### You would need to flip the operation from donor multiplication to donor addition with recipient as multiplication. Do a code review to understand this.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
from tqdm.auto import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================================
# GPU / perf setup
# ==========================================================
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

AMP_ENABLED = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16


# ==========================================================
# Config
# ==========================================================

@dataclass
class Config:
    p: int = 113
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    max_steps: int = 30000
    batch_size: int = 512
    curriculum_type: Literal["none", "complexity", "magnitude"] = "complexity"
    task_type: Literal["addition", "multiplication", "mixed"] = "mixed"
    curriculum_threshold: float = 0.95
    curriculum_window: int = 20
    log_interval: int = 250
    use_alibi: bool = True
    max_seq_len: int = 64
    save_dir: str = "final_results"
    experiment_name: str = "exp"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_set_size: int = 10000

    # --- Teacher-without-labels / embedding transplant probe ---
    transplant_embedding_path: Optional[str] = None
    freeze_transplanted_embedding: bool = False
    shuffle_donor_labels: bool = False

    # Opt-in: persist full model state_dict (~1.8MB/run). Needed ONLY for
    # runs that will later be reloaded for the component-ablation probe
    # (i.e. the mixed-task models). Everything else stays embedding-only.
    save_full_model: bool = False


# ==========================================================
# Model
# ==========================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool())
        if config.use_alibi:
            self.register_buffer("alibi_slopes", self._compute_alibi_slopes(config.n_heads))
        else:
            self.alibi_slopes = None

    @staticmethod
    def _compute_alibi_slopes(n_heads):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-8 / n)
                return [start * (start ** i) for i in range(n)]
            if n & (n - 1) == 0:
                return get_slopes_power_of_2(n)
            else:
                closest_power = 2 ** np.floor(np.log2(n))
                slopes = get_slopes_power_of_2(int(closest_power))
                slopes += get_slopes(int(2 * closest_power))[0::2][:n - int(closest_power)]
                return slopes
        return torch.tensor(get_slopes(n_heads), dtype=torch.float32)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        if self.alibi_slopes is not None:
            positions = torch.arange(T, device=x.device)
            alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().unsqueeze(0)
            att = att + alibi_bias.unsqueeze(0)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(nn.Linear(config.dim, 4 * config.dim), nn.GELU(), nn.Linear(4 * config.dim, config.dim), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.p = config.p
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self._init_weights()
        self._frozen_embedding_values = None  # set by _load_transplanted_embedding if freezing

        if config.transplant_embedding_path is not None:
            self._load_transplanted_embedding(config.transplant_embedding_path)
            if config.freeze_transplanted_embedding:
                self._freeze_numeric_embedding_rows()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_init)

    def _load_transplanted_embedding(self, path):
        donor_emb = np.load(path)
        assert donor_emb.shape[0] == self.p, f"donor embedding has {donor_emb.shape[0]} rows, expected {self.p}"
        assert donor_emb.shape[1] == self.token_emb.weight.shape[1], "donor embedding dim mismatch"
        with torch.no_grad():
            self.token_emb.weight[:self.p] = torch.tensor(
                donor_emb, dtype=self.token_emb.weight.dtype, device=self.token_emb.weight.device)
        print(f"  [transplant] loaded numeric-token embeddings from {path} (freeze={self.config.freeze_transplanted_embedding})")

        # CRITICAL: cache the exact frozen values so they can be hard-reset
        # after every optimizer step. Gradient-zeroing alone (see
        # _freeze_numeric_embedding_rows) is NOT sufficient: AdamW's
        # decoupled weight decay (param *= (1 - lr*weight_decay)) is applied
        # unconditionally to every parameter regardless of its gradient. At
        # weight_decay=1.0 with lr ~5e-4 (mid-cosine), this multiplies the
        # "frozen" embedding by ~0.9995 EVERY step, which compounds to
        # ~e^-15 (effectively zero) within a few thousand steps -- silently
        # destroying the transplanted structure via decay rather than
        # gradient updates. Hard-resetting the exact values after each
        # optimizer.step() neutralizes decay, momentum, and any other
        # optimizer side-effect on these rows, regardless of internals.
        if self.config.freeze_transplanted_embedding:
            self._frozen_embedding_values = self.token_emb.weight[:self.p].detach().clone()
        else:
            self._frozen_embedding_values = None

    def restore_frozen_rows(self):
        """Call after every optimizer.step() when freeze_transplanted_embedding
        is True, to undo any drift (weight decay, numerical, etc.) introduced
        by the optimizer despite the gradient being zeroed."""
        if self._frozen_embedding_values is not None:
            with torch.no_grad():
                self.token_emb.weight[:self.p] = self._frozen_embedding_values

    def _freeze_numeric_embedding_rows(self):
        """Freezes ONLY rows [0:p] via a backward hook (nn.Embedding has no
        partial-row requires_grad)."""
        numeric_rows = self.p

        def _zero_numeric_grad(grad):
            grad = grad.clone()
            grad[:numeric_rows] = 0
            return grad

        self.token_emb.weight.register_hook(_zero_numeric_grad)

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return F.linear(x, self.token_emb.weight[:self.p])

    def get_embedding_weights(self):
        return self.token_emb.weight[:self.p].detach().float().cpu().numpy()


# ==========================================================
# Curriculum
# ==========================================================

class Curriculum:
    def __init__(self, p, config):
        self.p = p
        self.config = config
        self.level = 0
        self.max_level = 2
        self.delim_token = p
        self.add_token = p + 1
        self.mul_token = p + 2
        self.vocab_size = p + 3
        self.accuracy_history = []

        if config.curriculum_type == "complexity":
            self.ranges = [(self.p // 2, self.p), (0, self.p), (0, self.p)]
        elif config.curriculum_type == "magnitude":
            self.ranges = [(0, self.p // 4), (0, self.p // 2), (0, self.p)]
        else:
            self.ranges = [(0, self.p)] * 3

        self.test_data = self._create_test_set()

    def _create_test_set(self):
        current_state = torch.get_rng_state()
        torch.manual_seed(999)
        size = self.config.test_set_size
        device = self.config.device
        a = torch.randint(0, self.p, (size,), device=device)
        b = torch.randint(0, self.p, (size,), device=device)
        if self.config.task_type == "addition":
            is_mul = torch.zeros(size, dtype=torch.bool, device=device)
        elif self.config.task_type == "multiplication":
            is_mul = torch.ones(size, dtype=torch.bool, device=device)
        else:
            is_mul = torch.rand(size, device=device) > 0.5
        ops = torch.where(is_mul, self.mul_token, self.add_token)
        y = torch.where(is_mul, (a * b) % self.p, (a + b) % self.p)
        delim = torch.full_like(a, self.delim_token)
        x = torch.stack([a, b, ops, delim], dim=1)
        torch.set_rng_state(current_state)
        return x, y

    def _sample_operands(self, batch_size, device):
        if self.config.curriculum_type == "none" or self.level >= self.max_level:
            low, high = 0, self.p
            use_mixed = False
        elif self.config.curriculum_type == "complexity" and self.level == 1:
            low, high = 0, self.p
            use_mixed = True
        else:
            low, high = self.ranges[self.level]
            use_mixed = False

        if use_mixed:
            n_full = int(batch_size * 0.7)
            n_wrap = batch_size - n_full
            a = torch.cat([torch.randint(0, self.p, (n_full,), device=device), torch.randint(self.p // 2, self.p, (n_wrap,), device=device)])
            b = torch.cat([torch.randint(0, self.p, (n_full,), device=device), torch.randint(self.p // 2, self.p, (n_wrap,), device=device)])
        else:
            a = torch.randint(low, high, (batch_size,), device=device)
            b = torch.randint(low, high, (batch_size,), device=device)
        return a, b

    def _labels(self, a, b, batch_size, device):
        if self.config.task_type == "addition":
            is_mul = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif self.config.task_type == "multiplication":
            is_mul = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_mul = torch.rand(batch_size, device=device) > 0.5
        ops = torch.where(is_mul, self.mul_token, self.add_token)
        y = torch.where(is_mul, (a * b) % self.p, (a + b) % self.p)
        return ops, y

    def get_batch(self, batch_size, mode='train'):
        if mode == 'test':
            indices = torch.randint(0, len(self.test_data[0]), (batch_size,))
            return self.test_data[0][indices], self.test_data[1][indices]
        device = self.config.device
        a, b = self._sample_operands(batch_size, device)
        ops, y = self._labels(a, b, batch_size, device)
        delim = torch.full_like(a, self.delim_token)
        x = torch.stack([a, b, ops, delim], dim=1)
        return x, y

    def update_progress(self, acc):
        if self.config.curriculum_type == "none" or self.level >= self.max_level:
            return False
        self.accuracy_history.append(acc)
        if len(self.accuracy_history) > self.config.curriculum_window:
            self.accuracy_history.pop(0)
        if len(self.accuracy_history) >= self.config.curriculum_window:
            if np.mean(self.accuracy_history) >= self.config.curriculum_threshold:
                self.level += 1
                self.accuracy_history.clear()
                return True
        return False


class ShuffledLabelCurriculum(Curriculum):
    """Negative control for donor training: identical operand sampling and
    curriculum schedule, but labels are permuted per-batch so no real
    (a, b) -> label structure can be learned. Isolates 'any pretraining
    helps' from 'structurally relevant pretraining helps'."""
    def get_batch(self, batch_size, mode='train'):
        x, y = super().get_batch(batch_size, mode=mode)
        if mode == 'train':
            perm = torch.randperm(y.shape[0], device=y.device)
            y = y[perm]
        return x, y


# ==========================================================
# Trainer
# ==========================================================

class Trainer:
    def __init__(self, model, curriculum, config):
        self.model = model
        self.curriculum = curriculum
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_steps, eta_min=config.learning_rate * 0.1)
        self.history = {'step': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'curriculum_level': [], 'generalization_gap': []}
        self.output_dir = Path(config.save_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_test(self):
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            x, y = self.curriculum.get_batch(1024, mode='test')
            logits = self.model(x)[:, -1, :]
            loss = F.cross_entropy(logits, y).item()
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
        self.model.train()
        return acc, loss

    def train(self, show_progress=True):
        pbar = tqdm(range(self.config.max_steps), desc=self.config.experiment_name, disable=not show_progress)
        train_grok = None
        test_grok = None

        for step in pbar:
            x, y = self.curriculum.get_batch(self.config.batch_size, mode='train')
            self.model.train()
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                logits = self.model(x)[:, -1, :]
                loss = F.cross_entropy(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.restore_frozen_rows()  # neutralize weight decay / drift on frozen rows -- see comment in _load_transplanted_embedding
            self.scheduler.step()

            if step % self.config.log_interval == 0:
                with torch.no_grad():
                    train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
                test_acc, test_loss = self.evaluate_test()
                gap = train_acc - test_acc
                self.history['step'].append(step)
                self.history['train_acc'].append(train_acc)
                self.history['test_acc'].append(test_acc)
                self.history['train_loss'].append(loss.item())
                self.history['test_loss'].append(test_loss)
                self.history['curriculum_level'].append(self.curriculum.level)
                self.history['generalization_gap'].append(gap)

                if train_acc >= 0.95 and train_grok is None:
                    train_grok = step
                if test_acc >= 0.95 and test_grok is None:
                    test_grok = step

                leveled_up = self.curriculum.update_progress(train_acc)
                if leveled_up and show_progress:
                    print(f"\n  Level {self.curriculum.level}")
                if show_progress:
                    pbar.set_postfix({'train': f'{train_acc:.3f}', 'test': f'{test_acc:.3f}', 'gap': f'{gap:.3f}', 'lvl': self.curriculum.level})

        results = {
            'train_grok_step': train_grok if train_grok is not None else self.config.max_steps,
            'test_grok_step': test_grok if test_grok is not None else self.config.max_steps,
            'train_grok_censored': train_grok is None,
            'test_grok_censored': test_grok is None,
            'final_train_acc': self.history['train_acc'][-1],
            'final_test_acc': self.history['test_acc'][-1],
            'final_gap': self.history['generalization_gap'][-1],
            'history': self.history,
            'config': {
                'curriculum': self.config.curriculum_type, 'task': self.config.task_type,
                'seed': self.config.seed, 'weight_decay': self.config.weight_decay,
                'max_steps': self.config.max_steps,
                'transplant_embedding_path': self.config.transplant_embedding_path,
                'freeze_transplanted_embedding': self.config.freeze_transplanted_embedding,
                'shuffle_donor_labels': self.config.shuffle_donor_labels,
            },
            'final_embeddings': self.model.get_embedding_weights(),
        }
        if self.config.save_full_model:
            results['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return results


# ==========================================================
# TEACHER-WITHOUT-LABELS TRANSPLANT PROBE
# ==========================================================

def train_donor_model(task, curriculum_type, seed, max_steps, shuffle_labels=False,
                       weight_decay=1.0, name=None, show_progress=True):
    """task: 'addition' or 'multiplication' -- the donor's own task, generalized
    so both transfer directions (mult->add and add->mult) can be tested with
    the same function."""
    name = name or f"donor_{task}_{'shuffled' if shuffle_labels else 'real'}_s{seed}"
    config = Config(curriculum_type=curriculum_type, task_type=task, seed=seed,
                     max_steps=max_steps, weight_decay=weight_decay,
                     experiment_name=name, shuffle_donor_labels=shuffle_labels)
    torch.manual_seed(seed)
    np.random.seed(seed)
    curriculum_cls = ShuffledLabelCurriculum if shuffle_labels else Curriculum
    curriculum_obj = curriculum_cls(config.p, config)
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    trainer = Trainer(model, curriculum_obj, config)
    result = trainer.train(show_progress=show_progress)

    out_dir = Path(config.save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['final_embeddings', 'model_state_dict']}, f, indent=2)
    emb_path = out_dir / 'final_embeddings.npy'
    np.save(emb_path, result['final_embeddings'])

    tag = 'shuffled' if shuffle_labels else 'real'
    print(f"  [donor:{task}:{tag}] saved {emb_path} (donor final_test_acc={result['final_test_acc']:.4f})")
    return result, str(emb_path)


def run_transplant_experiment(donor_embedding_path, freeze, seed, tag, recipient_task="addition",
                               max_steps=40000, weight_decay=1.0, show_progress=True):
    """recipient_task: the task the FRESH model (receiving the transplanted
    embedding) is trained on -- generalized so this can run either direction
    (donor=multiplication -> recipient_task='addition', or donor=addition ->
    recipient_task='multiplication')."""
    name = f"{tag}__recip-{recipient_task}__freeze{freeze}__s{seed}"
    config = Config(curriculum_type="complexity", task_type=recipient_task, seed=seed,
                     max_steps=max_steps, weight_decay=weight_decay, experiment_name=name,
                     transplant_embedding_path=donor_embedding_path,
                     freeze_transplanted_embedding=freeze)
    torch.manual_seed(seed)
    np.random.seed(seed)
    curriculum_obj = Curriculum(config.p, config)
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    trainer = Trainer(model, curriculum_obj, config)
    result = trainer.train(show_progress=show_progress)

    out_dir = Path(config.save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['final_embeddings', 'model_state_dict']}, f, indent=2)
    np.save(out_dir / 'final_embeddings.npy', result['final_embeddings'])

    result['tag'] = tag
    result['freeze'] = freeze
    result['recipient_task'] = recipient_task
    return result


def run_teacher_probe_suite(donor_task="multiplication", recipient_task="addition",
                             seeds=(42, 43, 44), donor_steps=40000, transplant_steps=40000,
                             show_progress=True):
    """Generalized to run EITHER transfer direction:
      - donor_task='multiplication', recipient_task='addition' (original direction)
      - donor_task='addition', recipient_task='multiplication' (reverse direction)
    Running both directions tests whether transfer is symmetric, or whether
    it's asymmetric in the same direction as the ablation-dissociation
    asymmetry found earlier (addition consistently more fragile than
    multiplication) -- i.e. does the FRAGILE task transfer worse as a
    RECIPIENT too, not just as an ablation target?
    """
    conditions = ['real_donor_frozen', 'real_donor_finetuned', 'shuffled_donor_frozen', 'shuffled_donor_finetuned']
    results = {c: [] for c in conditions}
    donor_paths = {'real': {}, 'shuffled': {}}

    for seed in seeds:
        print(f"\n=== Teacher probe ({donor_task} -> {recipient_task}): seed {seed} ===")
        print(f"[donor] training real-{donor_task} donor (seed {seed})")
        _, real_path = train_donor_model(donor_task, "complexity", seed, max_steps=donor_steps,
                                          shuffle_labels=False, show_progress=show_progress)
        donor_paths['real'][seed] = real_path

        print(f"[donor] training shuffled-label {donor_task} donor (seed {seed})")
        _, shuf_path = train_donor_model(donor_task, "complexity", seed, max_steps=donor_steps,
                                          shuffle_labels=True, show_progress=show_progress)
        donor_paths['shuffled'][seed] = shuf_path

        for freeze in (True, False):
            r_real = run_transplant_experiment(real_path, freeze, seed, "real_donor",
                                                recipient_task=recipient_task,
                                                max_steps=transplant_steps, show_progress=show_progress)
            r_shuf = run_transplant_experiment(shuf_path, freeze, seed, "shuffled_donor",
                                                recipient_task=recipient_task,
                                                max_steps=transplant_steps, show_progress=show_progress)
            key_real = 'real_donor_frozen' if freeze else 'real_donor_finetuned'
            key_shuf = 'shuffled_donor_frozen' if freeze else 'shuffled_donor_finetuned'
            results[key_real].append(r_real)
            results[key_shuf].append(r_shuf)

    return results, donor_paths


def summarize_teacher_probe(results: Dict[str, List[Dict]], direction_label: str = "") -> pd.DataFrame:
    rows = []
    for cond, runs in results.items():
        if not runs:
            continue
        accs = [r['final_test_acc'] for r in runs]
        groks = [r['test_grok_step'] for r in runs]
        censored = [r.get('test_grok_censored', False) for r in runs]
        rows.append({
            'condition': cond, 'direction': direction_label, 'n_seeds': len(runs),
            'mean_test_acc': np.mean(accs), 'std_test_acc': np.std(accs) if len(accs) > 1 else 0.0,
            'mean_grok_step': np.mean(groks), 'std_grok_step': np.std(groks) if len(groks) > 1 else 0.0,
            'frac_censored': np.mean(censored),
        })
    df = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_label = direction_label.replace(">", "to").replace(" ", "_") or "default"
    df.to_csv(save_dir / f"teacher_probe_summary_{safe_label}.csv", index=False)
    return df


def plot_bidirectional_comparison(summary_mult_to_add: pd.DataFrame, summary_add_to_mult: pd.DataFrame,
                                   save_dir: Path = Path("final_results/figures")):
    """The key rigor-improving figure: places both transfer directions
    side by side. Tests whether transfer is symmetric (same conditions
    succeed/fail regardless of direction) or asymmetric in a way that
    tracks the fragility asymmetry already found in the component-ablation
    probe (addition consistently more fragile than multiplication there).

    If real_donor_finetuned succeeds reliably in BOTH directions: transfer
    works regardless of which task is donor vs. recipient -- symmetric.
    If mult->add succeeds but add->mult is notably worse (or vice versa):
    directional asymmetry, and worth checking whether it lines up with
    which task was the more 'fragile' one in the ablation probe (there,
    addition was consistently the one that broke under ablation -- if
    addition is ALSO the harder task to transfer INTO here, that is a
    striking, consistent, doubly-confirmed asymmetry across two completely
    different experimental designs)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ['shuffled_donor_frozen', 'shuffled_donor_finetuned', 'real_donor_frozen', 'real_donor_finetuned']

    for ax, metric, ylabel, title in [
        (axes[0], 'mean_test_acc', 'Final Test Accuracy (mean +/- std)', 'Final Accuracy by Direction'),
    ]:
        pass  # placeholder to keep structure simple below

    df1 = summary_mult_to_add.set_index('condition').reindex(order).reset_index()
    df2 = summary_add_to_mult.set_index('condition').reindex(order).reset_index()

    x = np.arange(len(order))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, df1['mean_test_acc'], width, yerr=df1['std_test_acc'], capsize=5,
           label='mult -> add (original)', color='#2E86AB')
    ax.bar(x + width/2, df2['mean_test_acc'], width, yerr=df2['std_test_acc'], capsize=5,
           label='add -> mult (reverse)', color='#E67E22')
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Final Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title('Bidirectional Transfer: Final Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width/2, df1['mean_grok_step'], width, yerr=df1['std_grok_step'], capsize=5,
           label='mult -> add (original)', color='#2E86AB')
    ax.bar(x + width/2, df2['mean_grok_step'], width, yerr=df2['std_grok_step'], capsize=5,
           label='add -> mult (reverse)', color='#E67E22')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Steps to 95% Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title('Bidirectional Transfer: Grok Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'bidirectional_transfer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / 'bidirectional_transfer_comparison.png'}")

    print("\n=== Bidirectional asymmetry check ===")
    for cond in order:
        v1 = df1.loc[df1['condition'] == cond, 'mean_test_acc']
        v2 = df2.loc[df2['condition'] == cond, 'mean_test_acc']
        if len(v1) and len(v2):
            delta = float(v1.iloc[0]) - float(v2.iloc[0])
            print(f"  {cond}: mult->add={v1.iloc[0]:.1%}  add->mult={v2.iloc[0]:.1%}  "
                  f"delta={delta:+.1%} {'(mult->add higher)' if delta > 0.1 else '(add->mult higher)' if delta < -0.1 else '(comparable)'}")


def plot_teacher_probe(summary_df: pd.DataFrame, direction_label: str = "", save_dir: Path = Path("final_results/figures")):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ['shuffled_donor_frozen', 'shuffled_donor_finetuned', 'real_donor_frozen', 'real_donor_finetuned']
    df = summary_df.set_index('condition').reindex([c for c in order if c in summary_df['condition'].values]).reset_index()
    colors = ['#E74C3C', '#E67E22', '#95A5A6', '#2ECC71']
    title_suffix = f" ({direction_label})" if direction_label else ""

    ax = axes[0]
    bars = ax.bar(df['condition'], df['mean_test_acc'], yerr=df['std_test_acc'], capsize=6,
                   color=colors[:len(df)], edgecolor='black', linewidth=1.5)
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Final Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title(f'Teacher-Without-Labels Probe: Final Accuracy{title_suffix}', fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    for bar, m in zip(bars, df['mean_test_acc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{m:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(df['condition'], df['mean_grok_step'], yerr=df['std_grok_step'], capsize=6,
                   color=colors[:len(df)], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Steps to 95% Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title(f'Teacher-Without-Labels Probe: Grok Speed{title_suffix}', fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    for bar, m, c in zip(bars, df['mean_grok_step'], df['frac_censored']):
        label = f'{m:,.0f}' + (' (some censored)' if c > 0 else '')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    safe_label = direction_label.replace(">", "to").replace(" ", "_") or "default"
    plt.savefig(save_dir / f'teacher_probe_{safe_label}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Teacher probe figure saved to {save_dir / f'teacher_probe_{safe_label}.png'}")


# ==========================================================
# MIXED-TASK MODELS (needed for the component-ablation probe)
# ==========================================================

def train_mixed_task_model(seed, max_steps=40000, weight_decay=1.0, show_progress=True):
    """Trains a mixed-task (both operations, shared embedding) model WITH
    full weights saved -- required by the component-ablation probe below."""
    name = f"mixed_task__s{seed}"
    config = Config(curriculum_type="complexity", task_type="mixed", seed=seed,
                     max_steps=max_steps, weight_decay=weight_decay,
                     experiment_name=name, save_full_model=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    curriculum_obj = Curriculum(config.p, config)
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    trainer = Trainer(model, curriculum_obj, config)
    result = trainer.train(show_progress=show_progress)

    out_dir = Path(config.save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['final_embeddings', 'model_state_dict']}, f, indent=2)
    np.save(out_dir / 'final_embeddings.npy', result['final_embeddings'])
    torch.save(result['model_state_dict'], out_dir / 'model_state.pt')
    print(f"  [mixed-task s{seed}] final_test_acc={result['final_test_acc']:.4f}, saved to {out_dir}")

    return result, config, curriculum_obj, str(out_dir / 'model_state.pt')


# ==========================================================
# COMPONENT-ABLATION DISSOCIATION PROBE ("cognitive Legos" test)
# ==========================================================
#
# Motivated by the MIT PFC finding (Sur/Buschman 2026): the same neurons
# hold different task-relevant information at different moments, WITHOUT
# any change to synaptic weights -- flexible, weight-unchanged, context-
# gated reuse. The teacher-probe frozen-transplant result already tests
# (and rejects) the cross-model version of this analogy. This probe tests
# the WITHIN-model version: does a single grokked mixed-task model's shared
# embedding decompose into addition-specific / multiplication-specific /
# shared directions (a "Legos" structure), or is it fully entangled?
#
# Causal test: fit PCA on the trained embedding, ablate small groups of
# components, measure the DIFFERENTIAL damage to addition vs. multiplication
# accuracy. A random-direction ablation control (same rank) is included so
# PCA-targeted damage can be distinguished from generic fragility to any
# low-rank perturbation (the concern Xu 2026 raises with "transverse
# fragility").
# ==========================================================

def fit_embedding_pca(embedding: np.ndarray, n_components: int = 20) -> Dict:
    mean = embedding.mean(axis=0, keepdims=True)
    centered = embedding - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, Vt.shape[0])
    components = Vt[:n_components]
    explained_var = (S[:n_components] ** 2) / np.sum(S ** 2)
    return {'mean': mean, 'components': components, 'explained_variance_ratio': explained_var}


def ablate_components(embedding: np.ndarray, pca: Dict, component_indices: List[int]) -> np.ndarray:
    mean = pca['mean']
    centered = embedding - mean
    components = pca['components']
    coeffs = centered @ components.T
    coeffs_ablated = coeffs.copy()
    coeffs_ablated[:, component_indices] = 0.0
    reconstructed_centered = coeffs_ablated @ components
    residual = centered - (coeffs @ components)
    return mean + reconstructed_centered + residual


def ablate_random_directions(embedding: np.ndarray, n_directions: int, seed: int) -> np.ndarray:
    """Control: removes N random orthonormal directions (same rank as a PCA
    ablation group). Without this, 'ablating PCA components hurts accuracy'
    can't be distinguished from 'the model is fragile to ANY low-rank
    perturbation' -- a real concern given Xu (2026)'s transverse-fragility
    finding in a related setting."""
    rng = np.random.RandomState(seed)
    dim = embedding.shape[1]
    random_matrix = rng.randn(n_directions, dim)
    q, _ = np.linalg.qr(random_matrix.T)
    random_directions = q.T[:n_directions]
    mean = embedding.mean(axis=0, keepdims=True)
    centered = embedding - mean
    coeffs = centered @ random_directions.T
    projected_out = coeffs @ random_directions
    residual = centered - projected_out
    return mean + residual


def evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, embedding, task, n_eval=5000):
    device = config.device
    with torch.no_grad():
        model.token_emb.weight[:config.p] = torch.tensor(embedding, dtype=model.token_emb.weight.dtype, device=device)
    a = torch.randint(0, config.p, (n_eval,), device=device)
    b = torch.randint(0, config.p, (n_eval,), device=device)
    op_token = curriculum_obj.mul_token if task == 'multiplication' else curriculum_obj.add_token
    ops = torch.full_like(a, op_token)
    y = (a * b) % config.p if task == 'multiplication' else (a + b) % config.p
    delim = torch.full_like(a, curriculum_obj.delim_token)
    x = torch.stack([a, b, ops, delim], dim=1)
    model.eval()
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
    return acc


def run_component_ablation_experiment(config, curriculum_obj, model_state_path, seed,
                                       n_components=20, ablate_group_size=2, n_random_controls=3) -> pd.DataFrame:
    """Loads the ACTUAL trained mixed-task model (full weights, not just
    embedding), then runs PCA-group ablations + matched random-direction
    controls, measuring differential addition-vs-multiplication accuracy
    damage."""
    state_dict = torch.load(model_state_path, map_location='cpu')
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(config.device)

    baseline_embedding = model.get_embedding_weights()
    pca = fit_embedding_pca(baseline_embedding, n_components=n_components)

    rows = []
    baseline_add = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, baseline_embedding, 'addition')
    baseline_mul = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, baseline_embedding, 'multiplication')
    rows.append({'component_group': 'none (baseline)', 'ablation_type': 'none', 'addition_acc': baseline_add,
                 'multiplication_acc': baseline_mul, 'gap': baseline_add - baseline_mul})

    for start in range(0, n_components, ablate_group_size):
        idx = list(range(start, min(start + ablate_group_size, n_components)))
        ablated_embedding = ablate_components(baseline_embedding, pca, idx)
        add_acc = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, ablated_embedding, 'addition')
        mul_acc = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, ablated_embedding, 'multiplication')
        rows.append({'component_group': f"PC{idx[0]}-{idx[-1]}", 'ablation_type': 'pca',
                     'addition_acc': add_acc, 'multiplication_acc': mul_acc, 'gap': add_acc - mul_acc,
                     'explained_var_pct': float(np.sum(pca['explained_variance_ratio'][idx]) * 100)})

    for control_i in range(n_random_controls):
        control_seed = seed * 1000 + control_i
        random_ablated = ablate_random_directions(baseline_embedding, ablate_group_size, control_seed)
        add_acc = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, random_ablated, 'addition')
        mul_acc = evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, random_ablated, 'multiplication')
        rows.append({'component_group': f"random_control_{control_i}", 'ablation_type': 'random_control',
                     'addition_acc': add_acc, 'multiplication_acc': mul_acc, 'gap': add_acc - mul_acc})

    with torch.no_grad():
        model.token_emb.weight[:config.p] = torch.tensor(baseline_embedding, dtype=model.token_emb.weight.dtype, device=config.device)

    df = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / f"component_ablation_s{seed}.csv", index=False)

    pca_gaps = df[df['ablation_type'] == 'pca']['gap'].abs()
    random_gaps = df[df['ablation_type'] == 'random_control']['gap'].abs()
    verdict = 'PCA exceeds control' if pca_gaps.max() > random_gaps.max() else 'PCA does NOT exceed control -- likely generic fragility'
    print(f"  [seed {seed}] Max |gap| -- PCA-targeted: {pca_gaps.max():.3f} | random control: {random_gaps.max():.3f} ({verdict})")
    return df


def run_full_ablation_suite(seeds=(42, 43, 44), donor_steps=40000, n_components=20,
                             ablate_group_size=2, n_random_controls=3, show_progress=True) -> Dict[int, pd.DataFrame]:
    """Trains the mixed-task models needed (if not already trained) and runs
    the component-ablation dissociation test across seeds."""
    all_dfs = {}
    for seed in seeds:
        print(f"\n=== Mixed-task model + ablation probe: seed {seed} ===")
        result, config, curriculum_obj, state_path = train_mixed_task_model(
            seed, max_steps=donor_steps, show_progress=show_progress)
        df = run_component_ablation_experiment(config, curriculum_obj, state_path, seed,
                                                 n_components=n_components,
                                                 ablate_group_size=ablate_group_size,
                                                 n_random_controls=n_random_controls)
        all_dfs[seed] = df
    return all_dfs


def summarize_dissociation_across_seeds(all_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """CAVEAT: PCA is fit separately per seed -- 'PC0' in seed 42 is not the
    same direction as 'PC0' in seed 43 (independently trained models, PCA
    axes need not align, sign is arbitrary). We therefore aggregate only
    what's validly comparable across seeds: whether SOME dissociation
    reliably appears, and its magnitude relative to the random-direction
    control -- NOT which specific component index carries it."""
    rows = []
    for seed, df in all_dfs.items():
        pca_rows = df[df['ablation_type'] == 'pca'].copy()
        random_rows = df[df['ablation_type'] == 'random_control']
        if pca_rows.empty:
            continue
        pca_rows['abs_gap'] = pca_rows['gap'].abs()
        max_row = pca_rows.loc[pca_rows['abs_gap'].idxmax()]
        rows.append({
            'seed': seed,
            'max_abs_pca_gap': max_row['abs_gap'],
            'max_gap_component_group': max_row['component_group'],
            'max_gap_hurts': 'addition' if max_row['gap'] < 0 else 'multiplication',
            'max_abs_random_control_gap': random_rows['gap'].abs().max() if len(random_rows) else np.nan,
            'pca_exceeds_control': bool(max_row['abs_gap'] > (random_rows['gap'].abs().max() if len(random_rows) else 0)),
        })
    summary = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "dissociation_cross_seed_summary.csv", index=False)
    print("\n=== Cross-seed dissociation summary ===")
    print(summary.to_string(index=False))
    if len(summary) > 0:
        n_exceed = summary['pca_exceeds_control'].sum()
        print(f"\n{n_exceed}/{len(summary)} seeds: PCA-targeted ablation exceeds random-direction control damage.")
        print("If N/N with consistent max_gap_hurts direction: evidence for decomposable substructure.")
        print("If PCA rarely/never exceeds control: entangled representation, no 'Legos' structure detected "
              "(consistent with the frozen-transplant failure -- both point away from the naive MIT analogy).")
    return summary


def plot_dissociation_summary(summary: pd.DataFrame, save_dir: Path = Path("final_results/figures")):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary['max_abs_pca_gap'], width, label='PCA-targeted ablation', color='#2E86AB')
    ax.bar(x + width/2, summary['max_abs_random_control_gap'], width, label='Random-direction control', color='#95A5A6')
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in summary['seed']])
    ax.set_ylabel('Max |Addition Acc - Multiplication Acc|', fontsize=12, fontweight='bold')
    ax.set_title('PCA-Targeted vs Random-Direction Ablation Damage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'dissociation_cross_seed_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / 'dissociation_cross_seed_summary.png'}")


# ==========================================================
# EXTENDED-HORIZON FROZEN TRANSPLANT
# ==========================================================
#
# Resolves the last open question: is frozen-transplant failure genuine
# BASIS-MISMATCH (the donor embedding's coordinate system is fundamentally
# incompatible with what a fresh model's downstream blocks can ever learn
# to read), or just INSUFFICIENT ADAPTATION TIME (30K steps wasn't enough
# for the downstream attention/MLP blocks -- which were NEVER frozen -- to
# learn to interpret a fixed, wrong-task-shaped embedding)?
#
# Design: rerun the frozen conditions (real donor, shuffled donor) at a much
# larger step budget, REUSING the exact same donor embeddings already
# trained in run_teacher_probe_suite (no need to retrain donors -- transplant
# only ever needs the saved .npy, not the donor's own downstream weights).
#
# Interpretation:
#   - If frozen-real climbs substantially above the 30K result (e.g. toward
#     the fine-tuned-real ceiling) given enough steps: this was an
#     adaptation-TIME problem. The downstream blocks *can* learn to read
#     around a frozen embedding, just slowly. Softens the "basis-mismatch"
#     claim considerably.
#   - If frozen-real stays near-zero even at 3x+ the step budget, while
#     fine-tuned-real still reaches ~100% quickly: this is much stronger
#     evidence for genuine basis-mismatch -- the embedding coordinates
#     themselves are not just slow to read, they are not usable AT ALL
#     without directly modifying them. This is the result that would
#     upgrade "basis-alignment" from a speculative interpretation to an
#     evidenced one.
#   - Compare frozen-real vs frozen-shuffled at the extended horizon too: if
#     both remain equally near-zero, the failure is not even
#     content-dependent -- ANY frozen embedding (structured or not) is
#     unusable, which points toward a more basic architectural constraint
#     (e.g. the model's downstream blocks fundamentally require the
#     embedding itself to be optimizable, regardless of what it encodes).
# ==========================================================

def run_extended_frozen_transplant(donor_embedding_path, seed, tag, recipient_task="addition",
                                    max_steps=100000, weight_decay=1.0, show_progress=True):
    name = f"{tag}__recip-{recipient_task}__extended_frozen__s{seed}__steps{max_steps}"
    config = Config(curriculum_type="complexity", task_type=recipient_task, seed=seed,
                     max_steps=max_steps, weight_decay=weight_decay, experiment_name=name,
                     transplant_embedding_path=donor_embedding_path,
                     freeze_transplanted_embedding=True,
                     log_interval=500)  # coarser logging given the longer horizon
    torch.manual_seed(seed)
    np.random.seed(seed)
    curriculum_obj = Curriculum(config.p, config)
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    trainer = Trainer(model, curriculum_obj, config)
    result = trainer.train(show_progress=show_progress)

    out_dir = Path(config.save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['final_embeddings', 'model_state_dict']}, f, indent=2)
    np.save(out_dir / 'final_embeddings.npy', result['final_embeddings'])

    result['tag'] = tag
    print(f"  [extended-frozen] {tag} s{seed} @ {max_steps} steps: "
          f"final_test_acc={result['final_test_acc']:.4f}, censored={result.get('test_grok_censored', True)}")
    return result


def run_extended_frozen_suite(donor_paths: Dict[str, Dict[int, str]], recipient_task="addition",
                               seeds=(42, 43, 44), max_steps=100000, show_progress=True) -> Dict[str, List[Dict]]:
    results = {'real_extended_frozen': [], 'shuffled_extended_frozen': []}
    for seed in seeds:
        print(f"\n=== Extended-horizon frozen transplant (-> {recipient_task}): seed {seed} ({max_steps} steps) ===")
        r_real = run_extended_frozen_transplant(donor_paths['real'][seed], seed, "real_donor",
                                                  recipient_task=recipient_task,
                                                  max_steps=max_steps, show_progress=show_progress)
        results['real_extended_frozen'].append(r_real)

        r_shuf = run_extended_frozen_transplant(donor_paths['shuffled'][seed], seed, "shuffled_donor",
                                                  recipient_task=recipient_task,
                                                  max_steps=max_steps, show_progress=show_progress)
        results['shuffled_extended_frozen'].append(r_shuf)
    return results


def summarize_extended_frozen(extended_results: Dict[str, List[Dict]],
                               short_horizon_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Compares 30K-step frozen results (from the teacher probe) against the
    extended-horizon frozen results, so the basis-mismatch-vs-time question
    can be read directly off the table."""
    rows = []
    for label, runs in [('real_donor_frozen_30K', short_horizon_results.get('real_donor_frozen', [])),
                         ('shuffled_donor_frozen_30K', short_horizon_results.get('shuffled_donor_frozen', [])),
                         ('real_donor_frozen_extended', extended_results.get('real_extended_frozen', [])),
                         ('shuffled_donor_frozen_extended', extended_results.get('shuffled_extended_frozen', []))]:
        if not runs:
            continue
        accs = [r['final_test_acc'] for r in runs]
        steps = [r['config'].get('max_steps', np.nan) for r in runs]
        rows.append({
            'condition': label, 'n_seeds': len(runs), 'max_steps': steps[0] if steps else np.nan,
            'mean_test_acc': np.mean(accs), 'std_test_acc': np.std(accs) if len(accs) > 1 else 0.0,
        })
    df = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "extended_frozen_comparison.csv", index=False)
    print("\n=== Extended-horizon frozen transplant: basis-mismatch vs adaptation-time ===")
    print(df.to_string(index=False))

    real_30k = df.loc[df['condition'] == 'real_donor_frozen_30K', 'mean_test_acc']
    real_ext = df.loc[df['condition'] == 'real_donor_frozen_extended', 'mean_test_acc']
    if len(real_30k) and len(real_ext):
        delta = float(real_ext.iloc[0]) - float(real_30k.iloc[0])
        if delta > 0.3:
            print(f"\nreal-donor-frozen improved by {delta:.1%} with more steps -> ADAPTATION-TIME effect, "
                  f"not pure basis-mismatch. Soften any 'basis-alignment' claim accordingly.")
        elif delta < 0.05:
            print(f"\nreal-donor-frozen improved by only {delta:.1%} despite {max(df['max_steps']):.0f} steps -> "
                  f"consistent with genuine BASIS-MISMATCH, not just insufficient time. This is the evidence "
                  f"needed to make the basis-alignment claim directly rather than speculatively.")
        else:
            print(f"\nreal-donor-frozen improved by {delta:.1%} -- partial effect, report as-is without a strong claim either way.")
    return df


def plot_extended_frozen_comparison(df: pd.DataFrame, save_dir: Path = Path("final_results/figures")):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {'real_donor_frozen_30K': '#E74C3C', 'shuffled_donor_frozen_30K': '#F1948A',
              'real_donor_frozen_extended': '#2ECC71', 'shuffled_donor_frozen_extended': '#A9DFBF'}
    bar_colors = [colors.get(c, '#95A5A6') for c in df['condition']]
    bars = ax.bar(df['condition'], df['mean_test_acc'], yerr=df['std_test_acc'], capsize=6,
                   color=bar_colors, edgecolor='black', linewidth=1.5)
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Final Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title('Does Frozen Transfer Improve With More Steps?\n(Basis-Mismatch vs. Adaptation-Time)', fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    for bar, m in zip(bars, df['mean_test_acc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{m:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'extended_frozen_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / 'extended_frozen_comparison.png'}")

    # ==========================================================
# EMBEDDING REPRESENTATION VISUALIZATION
# ==========================================================
#
# For modular-arithmetic grokking, a well-generalized embedding typically
# organizes the p tokens along a circle (or similar periodic structure) in
# its top PCA components. Plotting this lets you SEE what the teacher-probe
# accuracy numbers only report indirectly: does the donor's circular
# structure (a) exist, (b) survive being copied into the recipient
# (frozen), and (c) get preserved/destroyed/reshaped during fine-tuning?

def plot_single_embedding_pca(ax, embedding: np.ndarray, title: str, p: int):
    """Projects embedding onto top-2 PCA axes, colors points 0..p-1 with a
    cyclic colormap (appropriate since these are values mod p)."""
    mean = embedding.mean(axis=0, keepdims=True)
    centered = embedding - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # (p, 2)
    var_explained = (S[:2] ** 2).sum() / (S ** 2).sum()

    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(p), cmap='hsv', s=25)
    ax.set_title(f"{title}\n(top-2 PC var: {var_explained:.1%})", fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2)
    return sc


def plot_embedding_grid(embeddings: Dict[str, np.ndarray], p: int, suptitle: str,
                         save_path: Path, ncols: int = 4):
    """embeddings: {label -> (p, dim) array}. Lays panels out in a grid,
    one shared colorbar for token value."""
    labels = list(embeddings.keys())
    n = len(labels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)

    sc = None
    for i, label in enumerate(labels):
        ax = axes[i // ncols][i % ncols]
        sc = plot_single_embedding_pca(ax, embeddings[label], label, p)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('token value (mod p)')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def collect_teacher_probe_embeddings(donor_paths: Dict[str, Dict[int, str]],
                                      transplant_results: Dict[str, List[Dict]],
                                      seed: int, direction_label: str) -> Dict[str, np.ndarray]:
    """Gathers, for ONE seed: the donor's own final embedding (real + shuffled),
    plus the recipient's final embedding under each of the 4 transplant
    conditions -- so donor and all recipients for that seed sit side by side."""
    out = {}
    # donor embeddings (loaded from the saved .npy, since donor result dicts
    # aren't kept around after train_donor_model returns)
    real_donor_path = Path(donor_paths['real'][seed])
    shuf_donor_path = Path(donor_paths['shuffled'][seed])
    out['donor (real)'] = np.load(real_donor_path)
    out['donor (shuffled)'] = np.load(shuf_donor_path)

    # recipient embeddings -- pull the matching-seed run out of each condition list
    cond_order = ['real_donor_frozen', 'real_donor_finetuned',
                  'shuffled_donor_frozen', 'shuffled_donor_finetuned']
    for cond in cond_order:
        for r in transplant_results.get(cond, []):
            if r['config']['seed'] == seed:
                out[f"recipient: {cond}"] = r['final_embeddings']
                break
    return out


def run_embedding_visualization_suite(donor_paths: Dict[str, Dict[int, str]],
                                       transplant_results: Dict[str, List[Dict]],
                                       p: int, direction_label: str,
                                       seeds=(42, 43, 44),
                                       save_dir: Path = Path("final_results/figures")):
    """One PCA grid per seed (donor real/shuffled + all 4 recipient conditions),
    so you can visually track: does the donor's circle survive frozen
    transplant? Does fine-tuning preserve it, reshape it, or destroy it?
    Does a shuffled (unstructured) donor ever produce a circle downstream?"""
    for seed in seeds:
        embeddings = collect_teacher_probe_embeddings(donor_paths, transplant_results, seed, direction_label)
        safe_label = direction_label.replace(">", "to").replace(" ", "_")
        plot_embedding_grid(
            embeddings, p,
            suptitle=f"Embedding structure across transplant conditions ({direction_label}, seed {seed})",
            save_path=save_dir / f"embedding_pca_{safe_label}_seed{seed}.png",
        )


# ==========================================================
# Main -- narrowed scope: teacher probe + component ablation ONLY
# ==========================================================

if __name__ == "__main__":
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | AMP enabled: {AMP_ENABLED}")

    # add -> mult was already run in a previous session, POST weight-decay
    # fix -- hardcoded here from that run's printed output rather than
    # re-executing it (saves ~2hr GPU time). If you rerun add->mult later,
    # replace this block with summarize_teacher_probe(probe_a2m, direction_label="add>mult").
    summary_a2m = pd.DataFrame([
        {'condition': 'real_donor_frozen',        'direction': 'add>mult', 'n_seeds': 3,
         'mean_test_acc': 0.999674, 'std_test_acc': 0.00046,
         'mean_grok_step': 25916.666667, 'std_grok_step': 1982.562876, 'frac_censored': 0.0},
        {'condition': 'real_donor_finetuned',      'direction': 'add>mult', 'n_seeds': 3,
         'mean_test_acc': 1.000000, 'std_test_acc': 0.00000,
         'mean_grok_step': 5250.000000, 'std_grok_step': 0.000000, 'frac_censored': 0.0},
        {'condition': 'shuffled_donor_frozen',     'direction': 'add>mult', 'n_seeds': 3,
         'mean_test_acc': 0.009115, 'std_test_acc': 0.00166,
         'mean_grok_step': 40000.000000, 'std_grok_step': 0.000000, 'frac_censored': 1.0},
        {'condition': 'shuffled_donor_finetuned',  'direction': 'add>mult', 'n_seeds': 3,
         'mean_test_acc': 1.000000, 'std_test_acc': 0.00000,
         'mean_grok_step': 6416.666667, 'std_grok_step': 772.801541, 'frac_censored': 0.0},
    ])
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_a2m.to_csv(save_dir / "teacher_probe_summary_add_to_mult.csv", index=False)
    print("Using previously-obtained (post-fix) add -> mult results (not rerun):")
    print(summary_a2m.to_string(index=False))


    # FRESH RUN, WITH THE WEIGHT-DECAY FIX: multiplication donor -> addition recipient
    # This is the critical test -- the OLD mult->add numbers (0.9% frozen-real)
    # were collected with the buggy code (frozen rows silently decayed to ~0
    # by AdamW weight_decay=1.0 despite the gradient-zeroing hook). This rerun
    # determines whether frozen-real success is symmetric across directions
    # or specific to add->mult.
    probe_m2a, donor_paths_m2a = run_teacher_probe_suite(
        donor_task="multiplication", recipient_task="addition", seeds=(42, 43, 44))
    summary_m2a = summarize_teacher_probe(probe_m2a, direction_label="mult>add")
    plot_teacher_probe(summary_m2a, direction_label="mult>add")
    run_embedding_visualization_suite(
    donor_paths_m2a, probe_m2a, p=Config().p,
    direction_label="mult>add", seeds=(42, 43, 44),
)
    print("\nmult -> add summary (POST-FIX):")
    print(summary_m2a.to_string(index=False))

    # Side-by-side bidirectional comparison, now BOTH directions post-fix
    plot_bidirectional_comparison(summary_m2a, summary_a2m)

    print("\nDone! (Component-ablation probe and extended-horizon frozen test "
          "can be run separately via run_full_ablation_suite() / "
          "run_extended_frozen_suite(donor_paths_m2a, recipient_task='addition', ...) "
          "using donor_paths_m2a returned above, if wanted -- note the extended-horizon "
          "results collected before the fix are also invalid and would need rerunning.)")
