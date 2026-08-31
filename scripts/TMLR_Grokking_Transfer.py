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
#
# CHANGE LOG (sumsquares refactor):
#   - task_type gains a "sumsquares" option: y = (a^2 + b^2) mod p.
#     This is a quadratic form, not a group homomorphism of the additive
#     OR multiplicative group structure of Z/pZ -- addition is linear,
#     multiplication becomes linear-in-angle under a DFT/log change of
#     basis (hence the known Fourier-circular embeddings for both), but
#     a^2+b^2 does not reduce to either. It is included specifically as a
#     NON-ISOMORPHIC third operator, to test whether the transfer/
#     dissociation asymmetries found between addition and multiplication
#     are general properties of "structural precision tolerance" or
#     artifacts of add/mult secretly sharing representational geometry.
#   - "mixed" now mixes across all task types present in `mixed_tasks`
#     (defaults to addition+multiplication, unchanged from before, but can
#     be widened to include sumsquares by passing mixed_tasks explicitly).
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
    task_type: Literal["addition", "multiplication", "sumsquares", "mixed"] = "mixed"
    # Which single-tasks "mixed" draws from. Only used when task_type == "mixed".
    # Kept as a tuple of strings (not a fancier type) to stay JSON-serializable
    # for the results dump.
    mixed_tasks: tuple = ("addition", "multiplication")
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
# Model  (UNCHANGED from original -- task type only affects data/labels,
# never model architecture, so nothing here needs to change to support
# sumsquares)
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
#
# CHANGE LOG (sumsquares refactor):
#   - Added self.sq_token (vocab_size grows by 1: p, p+1(add), p+2(mul),
#     p+3(sumsquares), delim moved to be assigned dynamically so token IDs
#     stay contiguous and unambiguous regardless of which tasks are active).
#   - _task_label(a, b, task) factors out the previously-inlined y=... logic
#     so addition/multiplication/sumsquares are computed by one function
#     instead of duplicated per-branch. This also makes it hard to forget to
#     add a new task in one of the two places (_create_test_set vs _labels)
#     that used to duplicate the same if/elif chain.
#   - _labels() and _create_test_set() now dispatch over an explicit list of
#     "active tasks" (config.task_type if single-task, else config.mixed_tasks
#     if task_type == "mixed"), rather than hardcoding is_mul as a boolean.
#     This generalizes cleanly to N task types instead of being wired for
#     exactly 2.
# ==========================================================

# Canonical task -> label function. Single source of truth so donor training,
# recipient training, the held-out test set, and the component-ablation probe
# can never disagree about what a given task means.
def compute_task_label(a: torch.Tensor, b: torch.Tensor, task: str, p: int) -> torch.Tensor:
    if task == "addition":
        return (a + b) % p
    elif task == "multiplication":
        return (a * b) % p
    elif task == "sumsquares":
        return (a.pow(2) + b.pow(2)) % p
    else:
        raise ValueError(f"Unknown task: {task}")


class Curriculum:
    # Fixed, deterministic token-id assignment for every task this codebase
    # knows about, so vocab layout never depends on which subset of tasks is
    # active in a given run (keeps donor/recipient token ids compatible with
    # each other regardless of task_type/mixed_tasks used in each).
    ALL_TASKS = ("addition", "multiplication", "sumsquares")

    def __init__(self, p, config):
        self.p = p
        self.config = config
        self.level = 0
        self.max_level = 2
        self.delim_token = p

        # Op tokens: p+1, p+2, p+3 (in ALL_TASKS order) -- always reserved,
        # regardless of which tasks are actually used in this run, so token
        # ids are stable across experiments/donors/recipients.
        self.op_token = {task: p + 1 + i for i, task in enumerate(self.ALL_TASKS)}
        self.add_token = self.op_token["addition"]
        self.mul_token = self.op_token["multiplication"]
        self.sq_token = self.op_token["sumsquares"]
        self.vocab_size = p + 1 + len(self.ALL_TASKS)

        # Which task(s) this run actually samples from.
        if config.task_type == "mixed":
            self.active_tasks = list(config.mixed_tasks)
        else:
            self.active_tasks = [config.task_type]
        assert all(t in self.ALL_TASKS for t in self.active_tasks), \
            f"active_tasks must be subset of {self.ALL_TASKS}, got {self.active_tasks}"

        self.accuracy_history = []

        if config.curriculum_type == "complexity":
            self.ranges = [(self.p // 2, self.p), (0, self.p), (0, self.p)]
        elif config.curriculum_type == "magnitude":
            self.ranges = [(0, self.p // 4), (0, self.p // 2), (0, self.p)]
        else:
            self.ranges = [(0, self.p)] * 3

        self.test_data = self._create_test_set()

    def _sample_active_task(self, batch_size, device):
        """Returns a LongTensor of length batch_size, values indexing into
        self.active_tasks, sampled uniformly. For single-task configs this
        is degenerate (always index 0) but keeping the same code path avoids
        a separate branch for single vs mixed."""
        n_tasks = len(self.active_tasks)
        if n_tasks == 1:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        return torch.randint(0, n_tasks, (batch_size,), device=device)

    def _labels_for_batch(self, a, b, task_idx, device):
        """task_idx: LongTensor mapping each example to an index into
        self.active_tasks. Computes both the op-token and the label per
        example by evaluating each active task's label fn and selecting,
        which generalizes the old is_mul boolean select to N tasks."""
        batch_size = a.shape[0]
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        ops = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, task in enumerate(self.active_tasks):
            mask = (task_idx == i)
            if not mask.any():
                continue
            y[mask] = compute_task_label(a[mask], b[mask], task, self.p)
            ops[mask] = self.op_token[task]
        return ops, y

    def _create_test_set(self):
        current_state = torch.get_rng_state()
        torch.manual_seed(999)
        size = self.config.test_set_size
        device = self.config.device
        a = torch.randint(0, self.p, (size,), device=device)
        b = torch.randint(0, self.p, (size,), device=device)
        task_idx = self._sample_active_task(size, device)
        ops, y = self._labels_for_batch(a, b, task_idx, device)
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
        task_idx = self._sample_active_task(batch_size, device)
        return self._labels_for_batch(a, b, task_idx, device)

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
# Trainer  (UNCHANGED from original)
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
                'mixed_tasks': list(self.config.mixed_tasks) if self.config.task_type == "mixed" else None,
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
# (unchanged logic -- generalizes automatically to sumsquares since
# task/donor_task/recipient_task are already plain strings threaded through)
# ==========================================================

def train_donor_model(task, curriculum_type, seed, max_steps, shuffle_labels=False,
                       weight_decay=1.0, name=None, show_progress=True):
    """task: 'addition', 'multiplication', or 'sumsquares' -- the donor's own
    task, generalized so all transfer directions (including the new
    sumsquares pairings) can be tested with the same function."""
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
    embedding) is trained on -- generalized so this can run any
    donor_task -> recipient_task pairing, e.g. multiplication -> sumsquares,
    sumsquares -> multiplication, sumsquares -> addition, etc."""
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
    """Generalized to run ANY transfer direction between two tasks in
    Curriculum.ALL_TASKS, e.g.:
      - donor_task='multiplication', recipient_task='addition'
      - donor_task='addition', recipient_task='multiplication'
      - donor_task='multiplication', recipient_task='sumsquares'   (NEW)
      - donor_task='sumsquares', recipient_task='multiplication'   (NEW)
    Running the sumsquares pairings tests whether the add<->mult asymmetry
    found earlier is a general 'structural precision tolerance' property or
    an artifact specific to add/mult sharing (secretly related) circular/
    Fourier representational geometry. sumsquares is a quadratic form, not a
    homomorphism of either group structure, making it a genuine non-
    isomorphic control.
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


def plot_bidirectional_comparison(summary_a: pd.DataFrame, summary_b: pd.DataFrame,
                                   label_a: str = "direction A", label_b: str = "direction B",
                                   filename: str = "bidirectional_transfer_comparison.png",
                                   save_dir: Path = Path("final_results/figures")):
    """Places two transfer directions side by side. Generalized (via
    label_a/label_b/filename args) so this same plotting fn can be reused
    for mult<->add, mult<->sumsquares, sumsquares<->add, etc. without
    duplicating the plotting code per pair."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ['shuffled_donor_frozen', 'shuffled_donor_finetuned', 'real_donor_frozen', 'real_donor_finetuned']

    df1 = summary_a.set_index('condition').reindex(order).reset_index()
    df2 = summary_b.set_index('condition').reindex(order).reset_index()

    x = np.arange(len(order))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, df1['mean_test_acc'], width, yerr=df1['std_test_acc'], capsize=5,
           label=label_a, color='#2E86AB')
    ax.bar(x + width/2, df2['mean_test_acc'], width, yerr=df2['std_test_acc'], capsize=5,
           label=label_b, color='#E67E22')
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
           label=label_a, color='#2E86AB')
    ax.bar(x + width/2, df2['mean_grok_step'], width, yerr=df2['std_grok_step'], capsize=5,
           label=label_b, color='#E67E22')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Steps to 95% Test Accuracy (mean +/- std)', fontsize=12, fontweight='bold')
    ax.set_title('Bidirectional Transfer: Grok Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / filename}")

    print(f"\n=== Bidirectional asymmetry check ({label_a} vs {label_b}) ===")
    for cond in order:
        v1 = df1.loc[df1['condition'] == cond, 'mean_test_acc']
        v2 = df2.loc[df2['condition'] == cond, 'mean_test_acc']
        if len(v1) and len(v2):
            delta = float(v1.iloc[0]) - float(v2.iloc[0])
            print(f"  {cond}: {label_a}={v1.iloc[0]:.1%}  {label_b}={v2.iloc[0]:.1%}  "
                  f"delta={delta:+.1%} {'(A higher)' if delta > 0.1 else '(B higher)' if delta < -0.1 else '(comparable)'}")


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
#
# CHANGE LOG (sumsquares refactor): mixed_tasks param added so the
# component-ablation "Legos" probe can also be run on 3-way mixed models
# (addition+multiplication+sumsquares), not just the original 2-way. The
# default keeps the ORIGINAL 2-way behavior unless the caller opts in, so
# existing addition-vs-multiplication ablation results remain reproducible
# unchanged.
# ==========================================================

def train_mixed_task_model(seed, max_steps=40000, weight_decay=1.0,
                            mixed_tasks=("addition", "multiplication"), show_progress=True):
    """Trains a mixed-task (2 or more operations, shared embedding) model
    WITH full weights saved -- required by the component-ablation probe
    below. mixed_tasks defaults to the original (addition, multiplication)
    pair; pass e.g. ("addition","multiplication","sumsquares") to extend the
    ablation-dissociation probe to a 3-way split."""
    tasks_tag = "-".join(mixed_tasks)
    name = f"mixed_task__{tasks_tag}__s{seed}"
    config = Config(curriculum_type="complexity", task_type="mixed", seed=seed,
                     max_steps=max_steps, weight_decay=weight_decay,
                     experiment_name=name, save_full_model=True,
                     mixed_tasks=tuple(mixed_tasks))
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
    print(f"  [mixed-task {tasks_tag} s{seed}] final_test_acc={result['final_test_acc']:.4f}, saved to {out_dir}")

    return result, config, curriculum_obj, str(out_dir / 'model_state.pt')


# ==========================================================
# COMPONENT-ABLATION DISSOCIATION PROBE ("cognitive Legos" test)
# ==========================================================
#
# CHANGE LOG (sumsquares refactor):
#   - evaluate_task_accuracy_with_embedding now looks up op_token and the
#     label function generically (via curriculum_obj.op_token[task] and
#     compute_task_label), instead of hardcoding an addition/multiplication
#     branch, so it works for any task in Curriculum.ALL_TASKS including
#     sumsquares.
#   - run_component_ablation_experiment takes a `tasks` tuple (default
#     unchanged: ("addition","multiplication")) instead of being hardwired
#     to exactly those two, and reports a gap matrix across however many
#     tasks are passed rather than a single addition-minus-multiplication
#     scalar. With 2 tasks this is byte-for-byte equivalent to the original
#     behavior (same 'gap' column semantics); with 3 tasks it additionally
#     reports pairwise gaps.
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
    """Generalized to any task in Curriculum.ALL_TASKS via
    curriculum_obj.op_token[task] and the shared compute_task_label fn,
    instead of a hardcoded addition/multiplication branch."""
    device = config.device
    with torch.no_grad():
        model.token_emb.weight[:config.p] = torch.tensor(embedding, dtype=model.token_emb.weight.dtype, device=device)
    a = torch.randint(0, config.p, (n_eval,), device=device)
    b = torch.randint(0, config.p, (n_eval,), device=device)
    op_token = curriculum_obj.op_token[task]
    ops = torch.full_like(a, op_token)
    y = compute_task_label(a, b, task, config.p)
    delim = torch.full_like(a, curriculum_obj.delim_token)
    x = torch.stack([a, b, ops, delim], dim=1)
    model.eval()
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
    return acc


def run_component_ablation_experiment(config, curriculum_obj, model_state_path, seed,
                                       tasks=("addition", "multiplication"),
                                       n_components=20, ablate_group_size=2, n_random_controls=3) -> pd.DataFrame:
    """Loads the ACTUAL trained mixed-task model (full weights, not just
    embedding), then runs PCA-group ablations + matched random-direction
    controls, measuring differential accuracy damage ACROSS however many
    tasks are passed in `tasks` (default: original addition-vs-multiplication
    behavior, unchanged). For a 2-task call this reproduces the original
    'gap' column (addition_acc - multiplication_acc) exactly; for 3 tasks
    (e.g. adding sumsquares) it additionally reports every pairwise gap so
    the dissociation structure of a 3-way mixed model can be read off
    directly."""
    state_dict = torch.load(model_state_path, map_location='cpu')
    model = GrokkingTransformer(config, curriculum_obj.vocab_size).to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(config.device)

    baseline_embedding = model.get_embedding_weights()
    pca = fit_embedding_pca(baseline_embedding, n_components=n_components)

    def accs_for(embedding):
        return {t: evaluate_task_accuracy_with_embedding(config, curriculum_obj, model, embedding, t) for t in tasks}

    def row_from_accs(component_group, ablation_type, accs, extra=None, component_indices=None):
        row = {'component_group': component_group, 'ablation_type': ablation_type}
        for t in tasks:
            row[f'{t}_acc'] = accs[t]
        # Preserve the original 2-task 'gap' column name/semantics when
        # exactly 2 tasks are used (addition_acc - multiplication_acc, or
        # more generally tasks[0]_acc - tasks[1]_acc) for backward
        # compatibility with existing analysis/plotting code.
        if len(tasks) == 2:
            row['gap'] = accs[tasks[0]] - accs[tasks[1]]
        else:
            # 3+ tasks: report every pairwise gap explicitly instead of a
            # single ambiguous scalar.
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    row[f'gap_{tasks[i]}_minus_{tasks[j]}'] = accs[tasks[i]] - accs[tasks[j]]
        # NEW: persist the raw component indices (as a JSON string, since
        # CSV cells must be scalar) for every PCA ablation row. Previously
        # only the human-readable 'PC{start}-{end}' label was stored, which
        # is fine for plotting but cannot be reliably parsed back into an
        # index list downstream (e.g. if ablate_group_size ever varies, or
        # indices are non-contiguous). This is required by the new subspace-
        # alignment analysis below, which needs to reconstruct exactly which
        # PCA basis rows correspond to each ablation group. Purely additive:
        # does not change any existing column's meaning.
        row['component_indices'] = json.dumps(component_indices) if component_indices is not None else json.dumps([])
        if extra:
            row.update(extra)
        return row

    rows = []
    baseline_accs = accs_for(baseline_embedding)
    rows.append(row_from_accs('none (baseline)', 'none', baseline_accs, component_indices=[]))

    for start in range(0, n_components, ablate_group_size):
        idx = list(range(start, min(start + ablate_group_size, n_components)))
        ablated_embedding = ablate_components(baseline_embedding, pca, idx)
        accs = accs_for(ablated_embedding)
        rows.append(row_from_accs(f"PC{idx[0]}-{idx[-1]}", 'pca', accs,
                                    extra={'explained_var_pct': float(np.sum(pca['explained_variance_ratio'][idx]) * 100)},
                                    component_indices=idx))

    for control_i in range(n_random_controls):
        control_seed = seed * 1000 + control_i
        random_ablated = ablate_random_directions(baseline_embedding, ablate_group_size, control_seed)
        accs = accs_for(random_ablated)
        rows.append(row_from_accs(f"random_control_{control_i}", 'random_control', accs, component_indices=[]))

    with torch.no_grad():
        model.token_emb.weight[:config.p] = torch.tensor(baseline_embedding, dtype=model.token_emb.weight.dtype, device=config.device)

    df = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    tasks_tag = "-".join(tasks)
    df.to_csv(save_dir / f"component_ablation_{tasks_tag}_s{seed}.csv", index=False)

    # NEW: persist the fitted PCA basis itself (mean + component rows), not
    # just the CSV of ablation results. The subspace-alignment analysis
    # needs the actual orthonormal basis vectors -- these were previously
    # computed inside this function and then discarded once it returned,
    # meaning there was no way to reuse them later without retraining +
    # re-fitting PCA identically (fragile: any code change to fit_embedding_pca
    # or its inputs could silently produce a different basis). Saving them
    # directly removes that fragility.
    np.savez(save_dir / f"component_ablation_pca_{tasks_tag}_s{seed}.npz",
             mean=pca['mean'], components=pca['components'],
             explained_variance_ratio=pca['explained_variance_ratio'])

    gap_cols = [c for c in df.columns if c == 'gap' or c.startswith('gap_')]
    for gap_col in gap_cols:
        pca_gaps = df[df['ablation_type'] == 'pca'][gap_col].abs()
        random_gaps = df[df['ablation_type'] == 'random_control'][gap_col].abs()
        verdict = 'PCA exceeds control' if pca_gaps.max() > random_gaps.max() else 'PCA does NOT exceed control -- likely generic fragility'
        print(f"  [seed {seed}] [{gap_col}] Max |gap| -- PCA-targeted: {pca_gaps.max():.3f} | random control: {random_gaps.max():.3f} ({verdict})")
    return df


def run_full_ablation_suite(seeds=(42, 43, 44), donor_steps=40000,
                             mixed_tasks=("addition", "multiplication"),
                             n_components=20, ablate_group_size=2, n_random_controls=3,
                             show_progress=True) -> Dict[int, pd.DataFrame]:
    """Trains the mixed-task models needed (if not already trained) and runs
    the component-ablation dissociation test across seeds. mixed_tasks
    defaults to the original 2-way split; pass a 3-tuple including
    'sumsquares' to extend the probe."""
    all_dfs = {}
    for seed in seeds:
        print(f"\n=== Mixed-task model + ablation probe ({'/'.join(mixed_tasks)}): seed {seed} ===")
        result, config, curriculum_obj, state_path = train_mixed_task_model(
            seed, max_steps=donor_steps, mixed_tasks=mixed_tasks, show_progress=show_progress)
        df = run_component_ablation_experiment(config, curriculum_obj, state_path, seed,
                                                 tasks=mixed_tasks,
                                                 n_components=n_components,
                                                 ablate_group_size=ablate_group_size,
                                                 n_random_controls=n_random_controls)
        all_dfs[seed] = df
    return all_dfs


def summarize_dissociation_across_seeds(all_dfs: Dict[int, pd.DataFrame],
                                          tasks=("addition", "multiplication")) -> pd.DataFrame:
    """CAVEAT: PCA is fit separately per seed -- 'PC0' in seed 42 is not the
    same direction as 'PC0' in seed 43 (independently trained models, PCA
    axes need not align, sign is arbitrary). We therefore aggregate only
    what's validly comparable across seeds: whether SOME dissociation
    reliably appears, and its magnitude relative to the random-direction
    control -- NOT which specific component index carries it.

    Generalized to N tasks: uses whichever gap column(s) are present in the
    dataframe (either 'gap' for the 2-task case, or 'gap_<a>_minus_<b>' for
    each pair when 3+ tasks were ablated)."""
    gap_cols = None
    rows = []
    for seed, df in all_dfs.items():
        if gap_cols is None:
            gap_cols = [c for c in df.columns if c == 'gap' or c.startswith('gap_')]
        pca_rows = df[df['ablation_type'] == 'pca'].copy()
        random_rows = df[df['ablation_type'] == 'random_control']
        if pca_rows.empty:
            continue
        row = {'seed': seed}
        for gap_col in gap_cols:
            pca_rows[f'abs_{gap_col}'] = pca_rows[gap_col].abs()
            max_idx = pca_rows[f'abs_{gap_col}'].idxmax()
            max_row = pca_rows.loc[max_idx]
            random_max = random_rows[gap_col].abs().max() if len(random_rows) else np.nan
            row[f'max_abs_pca_{gap_col}'] = max_row[f'abs_{gap_col}']
            row[f'max_{gap_col}_component_group'] = max_row['component_group']
            row[f'max_abs_random_control_{gap_col}'] = random_max
            row[f'pca_exceeds_control_{gap_col}'] = bool(max_row[f'abs_{gap_col}'] > (random_max if not np.isnan(random_max) else 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_dir = Path("final_results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    tasks_tag = "-".join(tasks)
    summary.to_csv(save_dir / f"dissociation_cross_seed_summary_{tasks_tag}.csv", index=False)
    print(f"\n=== Cross-seed dissociation summary ({'/'.join(tasks)}) ===")
    print(summary.to_string(index=False))
    return summary


def plot_dissociation_summary(summary: pd.DataFrame, gap_col: str = 'gap',
                               filename: str = 'dissociation_cross_seed_summary.png',
                               save_dir: Path = Path("final_results/figures")):
    """gap_col: base gap column name used when the summary was built (e.g.
    'gap' for a 2-task ablation run). Reads the matching
    max_abs_pca_{gap_col} / max_abs_random_control_{gap_col} columns."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary[f'max_abs_pca_{gap_col}'], width, label='PCA-targeted ablation', color='#2E86AB')
    ax.bar(x + width/2, summary[f'max_abs_random_control_{gap_col}'], width, label='Random-direction control', color='#95A5A6')
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in summary['seed']])
    ax.set_ylabel(f'Max |{gap_col}|', fontsize=12, fontweight='bold')
    ax.set_title('PCA-Targeted vs Random-Direction Ablation Damage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / filename}")


# ==========================================================
# SUBSPACE-ALIGNMENT ANALYSIS: linking ablation-decomposability to
# transfer-asymmetry (mechanistic account for Claim A)
# ==========================================================
#
# MOTIVATION
# ----------
# Two causal results currently sit side by side without being connected:
#
#   (1) Component-ablation probe: within a SINGLE mixed-task model, the
#       shared embedding decomposes into addition-relevant and
#       multiplication-relevant low-rank subspaces (PCA-targeted ablation
#       damage exceeds a random-direction control). This establishes that
#       task-relevant structure is LOCALIZED / DECOMPOSABLE.
#
#   (2) Teacher-probe transplant suite: transplanting a donor's embedding
#       into a FRESH model learning a different task shows a directional
#       ASYMMETRY (e.g. shuffled-donor-finetuned fully rescues one transfer
#       direction but only partially rescues the reverse direction).
#
# Neither result alone explains the other. (1) says structure is
# decomposable; it says nothing about why transplanting it in one direction
# works better than the other. (2) says transfer is asymmetric; it says
# nothing about WHY, mechanistically.
#
# This section closes that gap with a single, concrete, testable link:
#
#   HYPOTHESIS: a donor's OWN task-specific subspace (fit directly on a
#   standalone donor trained only on that task) should be geometrically
#   ALIGNED with the corresponding task-relevant subspace identified inside
#   the mixed-task model (via ablation). If addition's dedicated subspace
#   and multiplication's dedicated subspace are themselves poorly aligned
#   with EACH OTHER (i.e. genuinely distinct directions in embedding
#   space), that predicts asymmetric/fragile transfer between them BETTER
#   than treating "structure" as a single undifferentiated concept.
#   Furthermore, if the CROSS alignment (e.g. the multiplication donor's
#   subspace vs. the mixed-model's addition-relevant subspace) differs in
#   magnitude depending on direction, that is a direct geometric candidate
#   explanation for the observed transfer asymmetry, rather than an
#   unexplained empirical curiosity.
#
# METHOD
# ------
# Subspace alignment is measured via principal angles (a standard, basis-
# independent way to compare two linear subspaces of a shared ambient
# space): given orthonormal bases A (k1 x dim) and B (k2 x dim), the
# singular values of A @ B.T are the cosines of the principal angles
# between the subspaces they span. A mean cosine near 1 means the
# subspaces are nearly coincident; near 0 means they are nearly orthogonal.
# This is preferred over naive per-vector cosine similarity because PCA
# components are only defined up to sign/rotation within a subspace of
# similar eigenvalue -- principal angles are invariant to that ambiguity.
# ==========================================================

def principal_angle_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """basis_a: (k1, dim) orthonormal rows (e.g. PCA components).
    basis_b: (k2, dim) orthonormal rows.
    Returns the min(k1,k2) singular values of basis_a @ basis_b.T, which are
    the cosines of the principal angles between the two subspaces (each in
    [0, 1], since PCA components are already orthonormal; clipped defensively
    in case of numerical drift)."""
    # Re-orthonormalize defensively -- PCA components from np.linalg.svd
    # should already be orthonormal, but this guards against any upstream
    # change (e.g. a future basis source that isn't perfectly orthonormal).
    def _orthonormalize(basis):
        q, _ = np.linalg.qr(basis.T)
        return q.T[:basis.shape[0]]
    a = _orthonormalize(basis_a)
    b = _orthonormalize(basis_b)
    m = a @ b.T
    cosines = np.linalg.svd(m, compute_uv=False)
    return np.clip(cosines, -1.0, 1.0)


def identify_task_relevant_components(ablation_df: pd.DataFrame, task_a: str, task_b: str,
                                       top_frac: float = 0.3) -> Dict[str, List[int]]:
    """Given a component-ablation results dataframe (with the new
    'component_indices' column) for exactly the (task_a, task_b) pair used
    when it was generated, returns the PCA component indices most
    responsible for damaging EACH task specifically.

    Uses the 'gap' column (= task_a_acc - task_b_acc, per
    run_component_ablation_experiment's convention when len(tasks)==2).
    Ablation groups with the most NEGATIVE gap (task_a accuracy fell far
    below task_b) are task_a-relevant; groups with the most POSITIVE gap
    are task_b-relevant. top_frac controls how many of the most extreme
    groups (by count of ablation groups, not by variance) are pooled into
    each task's subspace -- e.g. top_frac=0.3 with 10 ablation groups pools
    the 3 most negative-gap groups' indices for task_a and the 3 most
    positive-gap groups' indices for task_b.

    Returns duplicate-free, sorted index lists; note the two lists CAN
    overlap if the same component group is highly damaging to both
    directions in different ablation groups -- this is reported, not
    silently resolved, since overlap is itself informative (it would argue
    against clean decomposability)."""
    pca_rows = ablation_df[ablation_df['ablation_type'] == 'pca'].copy()
    if 'gap' not in pca_rows.columns:
        raise ValueError("identify_task_relevant_components currently only supports the 2-task "
                         "'gap' column convention (len(tasks)==2 when the ablation was run). "
                         "For 3+ tasks, adapt to use the appropriate gap_{a}_minus_{b} column.")
    pca_rows['idx_list'] = pca_rows['component_indices'].apply(json.loads)
    pca_rows = pca_rows.sort_values('gap')
    n_groups = len(pca_rows)
    n_pool = max(1, int(np.ceil(n_groups * top_frac)))

    task_a_rows = pca_rows.iloc[:n_pool]  # most negative gap -> hurts task_a most
    task_b_rows = pca_rows.iloc[-n_pool:]  # most positive gap -> hurts task_b most

    task_a_indices = sorted(set(i for lst in task_a_rows['idx_list'] for i in lst))
    task_b_indices = sorted(set(i for lst in task_b_rows['idx_list'] for i in lst))
    overlap = sorted(set(task_a_indices) & set(task_b_indices))
    if overlap:
        print(f"  [identify_task_relevant_components] WARNING: {len(overlap)} component index/indices "
              f"({overlap}) appear in BOTH {task_a}-relevant and {task_b}-relevant pools -- "
              f"subspaces are not cleanly disjoint at top_frac={top_frac}.")

    return {task_a: task_a_indices, task_b: task_b_indices}


def load_ablation_pca_basis(tasks: tuple, seed: int, save_dir: Path = Path("final_results/figures")) -> Dict:
    """Loads the PCA basis (.npz) saved by run_component_ablation_experiment
    for a given task pair + seed. Must be called AFTER that function has
    been run with the same tasks/seed (raises FileNotFoundError otherwise,
    with a clear message rather than a cryptic numpy error)."""
    tasks_tag = "-".join(tasks)
    path = save_dir / f"component_ablation_pca_{tasks_tag}_s{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_component_ablation_experiment(tasks={tasks}, seed={seed}, ...) "
            f"first -- it saves this file as a side effect.")
    data = np.load(path)
    return {'mean': data['mean'], 'components': data['components'],
            'explained_variance_ratio': data['explained_variance_ratio']}


def fit_donor_task_subspace(donor_embedding_path: str, n_components: int) -> np.ndarray:
    """Loads a standalone donor's final embedding (trained ONLY on its own
    task, e.g. via train_donor_model) and fits PCA on it, returning the top
    n_components as an orthonormal basis. This is the donor's OWN
    task-specific subspace, independent of any mixed-task model -- the
    comparison point for alignment against the mixed-model's ablation-
    identified task-relevant subspace."""
    embedding = np.load(donor_embedding_path)
    pca = fit_embedding_pca(embedding, n_components=n_components)
    return pca['components']


def run_subspace_alignment_analysis(seed: int, tasks: tuple,
                                     donor_embedding_paths: Dict[str, str],
                                     top_frac: float = 0.3,
                                     save_dir: Path = Path("final_results/figures")) -> pd.DataFrame:
    """Core mechanistic-linkage analysis for Claim A.

    tasks: the exact 2-tuple used when run_component_ablation_experiment was
        called for this seed (e.g. ("addition", "multiplication")).
    donor_embedding_paths: dict mapping each task in `tasks` to the path of
        a STANDALONE donor's final_embeddings.npy (i.e. the 'real donor'
        embeddings already produced by train_donor_model / the teacher-probe
        suite for this same seed -- reuse those, do not retrain).

    Returns a DataFrame with one row per (mixed_model_task_subspace,
    donor_task_subspace) pair, reporting the mean cosine of principal
    angles between them. The diagonal (task X's mixed-model subspace vs.
    task X's own donor subspace) should be HIGH if ablation is correctly
    identifying real task-specific structure. The off-diagonal terms are
    the ones that matter for explaining transfer asymmetry: e.g. if
    multiplication's mixed-model subspace is well-aligned with the
    ADDITION donor's subspace, but addition's mixed-model subspace is
    poorly aligned with the MULTIPLICATION donor's subspace, that
    asymmetry is a direct geometric candidate explanation for why
    mult->add and add->mult transfer differently.
    """
    assert len(tasks) == 2, "This analysis currently supports exactly 2 tasks (matches the 'gap' column convention)."
    task_a, task_b = tasks

    tasks_tag = "-".join(tasks)
    ablation_csv_path = save_dir / f"component_ablation_{tasks_tag}_s{seed}.csv"
    if not ablation_csv_path.exists():
        raise FileNotFoundError(
            f"{ablation_csv_path} not found. Run run_component_ablation_experiment(tasks={tasks}, "
            f"seed={seed}, ...) first.")
    ablation_df = pd.read_csv(ablation_csv_path)

    relevant = identify_task_relevant_components(ablation_df, task_a, task_b, top_frac=top_frac)
    pca_basis = load_ablation_pca_basis(tasks, seed, save_dir=save_dir)['components']

    mixed_subspace = {}
    for task in tasks:
        idx = relevant[task]
        if len(idx) == 0:
            raise ValueError(f"No ablation-identified components found for task '{task}' at "
                             f"top_frac={top_frac} -- try increasing top_frac.")
        mixed_subspace[task] = pca_basis[idx]

    donor_subspace = {}
    for task in tasks:
        k = mixed_subspace[task].shape[0]
        donor_subspace[task] = fit_donor_task_subspace(donor_embedding_paths[task], n_components=k)

    rows = []
    for mixed_task in tasks:
        for donor_task in tasks:
            cosines = principal_angle_cosines(mixed_subspace[mixed_task], donor_subspace[donor_task])
            rows.append({
                'mixed_model_subspace': mixed_task,
                'donor_subspace': donor_task,
                'mean_cosine': float(np.mean(cosines)),
                'min_cosine': float(np.min(cosines)),
                'n_dims_compared': int(len(cosines)),
                'relationship': 'same-task (expect high)' if mixed_task == donor_task else 'cross-task (asymmetry probe)',
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / f"subspace_alignment_{tasks_tag}_s{seed}.csv", index=False)
    print(f"\n=== Subspace alignment ({'/'.join(tasks)}), seed {seed} ===")
    print(df.to_string(index=False))
    return df


def summarize_subspace_alignment_across_seeds(tasks: tuple, seeds: tuple,
                                                save_dir: Path = Path("final_results/figures")) -> pd.DataFrame:
    """Aggregates run_subspace_alignment_analysis results across seeds
    (mean +/- std of mean_cosine per (mixed_model_subspace, donor_subspace)
    cell), analogous in spirit to summarize_teacher_probe /
    summarize_dissociation_across_seeds elsewhere in this file."""
    tasks_tag = "-".join(tasks)
    all_dfs = []
    for seed in seeds:
        path = save_dir / f"subspace_alignment_{tasks_tag}_s{seed}.csv"
        if not path.exists():
            print(f"  [warn] missing {path}, skipping seed {seed}")
            continue
        d = pd.read_csv(path)
        d['seed'] = seed
        all_dfs.append(d)
    if not all_dfs:
        raise FileNotFoundError(f"No subspace_alignment CSVs found for tasks={tasks}, seeds={seeds}. "
                                f"Run run_subspace_alignment_analysis for each seed first.")
    combined = pd.concat(all_dfs, ignore_index=True)
    summary = combined.groupby(['mixed_model_subspace', 'donor_subspace', 'relationship']).agg(
        mean_cosine_avg=('mean_cosine', 'mean'),
        mean_cosine_std=('mean_cosine', lambda s: np.std(s) if len(s) > 1 else 0.0),
        n_seeds=('mean_cosine', 'count'),
    ).reset_index()
    summary.to_csv(save_dir / f"subspace_alignment_summary_{tasks_tag}.csv", index=False)
    print(f"\n=== Subspace alignment summary across seeds ({'/'.join(tasks)}) ===")
    print(summary.to_string(index=False))
    return summary


def plot_subspace_alignment_heatmap(summary: pd.DataFrame, tasks: tuple,
                                     transfer_asymmetry_note: Optional[str] = None,
                                     save_dir: Path = Path("final_results/figures")):
    """THE key mechanistic figure linking Claim A's two causal results.

    Renders a (mixed_model_subspace x donor_subspace) heatmap of mean
    principal-angle cosine (alignment), annotated with values, so the
    reader can see at a glance:
      - whether the diagonal (same-task) alignment is high (validates that
        ablation is finding real, donor-matching structure), and
      - whether the off-diagonal (cross-task) alignment is ASYMMETRIC in a
        way that lines up with the observed transfer asymmetry (e.g. if
        cell [mult-subspace, add-donor] >> cell [add-subspace, mult-donor],
        that is a geometric candidate explanation for why one transfer
        direction succeeds more easily than the other).

    transfer_asymmetry_note: optional short string (e.g.
        "Observed: shuffled-donor-finetuned reaches 100% add->mult but "
        "only ~35% mult->add") printed as a figure subtitle/annotation so
        the geometric result and the behavioral result sit side by side
        for the reader without requiring them to cross-reference two
        separate figures.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    task_order = list(tasks)
    matrix = summary.pivot(index='mixed_model_subspace', columns='donor_subspace', values='mean_cosine_avg')
    matrix = matrix.reindex(index=task_order, columns=task_order)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    im = ax.imshow(matrix.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(task_order)))
    ax.set_yticks(range(len(task_order)))
    ax.set_xticklabels([f"{t}\n(donor subspace)" for t in task_order], fontsize=10)
    ax.set_yticklabels([f"{t}\n(mixed-model subspace)" for t in task_order], fontsize=10)
    for i in range(len(task_order)):
        for j in range(len(task_order)):
            val = matrix.values[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val < 0.6 else 'black'
            marker = ' (same-task)' if task_order[i] == task_order[j] else ' (cross-task)'
            ax.text(j, i, f"{val:.2f}{marker}", ha='center', va='center', fontsize=10,
                    fontweight='bold', color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean principal-angle cosine (subspace alignment)', fontsize=10, fontweight='bold')

    title = 'Subspace Alignment: Ablation-Identified Structure vs. Standalone Donor Structure'
    if transfer_asymmetry_note:
        title += f"\n{transfer_asymmetry_note}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"subspace_alignment_heatmap_{'-'.join(tasks)}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / fname}")


# ==========================================================
# EXTENDED-HORIZON FROZEN TRANSPLANT  (unchanged logic; already generic
# over recipient_task string)
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
# NEW: PILOT RUN for sumsquares
# ==========================================================
#
# Purpose: before committing to the full 5-seed x 2x2x2 transplant grid for
# sumsquares<->multiplication (and/or sumsquares<->addition), run a cheap
# single-seed check that sumsquares groks cleanly on its own (no transplant),
# with and without curriculum, comparable to the existing operation-dependent
# grokking-speed figure for addition/multiplication. This directly guards
# against the confound flagged earlier: if sumsquares has a very different
# intrinsic difficulty (plausible, since squaring mod a prime is 2-to-1 --
# roughly half of Z/pZ are quadratic residues), any later transplant
# asymmetry could be explained by "different intrinsic difficulty" rather
# than "non-isomorphic representational structure", so this needs to be
# checked and reported BEFORE the full suite is interpreted.
# ==========================================================

def run_sumsquares_pilot(seed=42, max_steps=30000, show_progress=True) -> Dict[str, Dict]:
    """Runs sumsquares alone (single-task, no transplant) both with the
    complexity curriculum and with no curriculum, mirroring the
    curriculum-necessity check already done for addition/multiplication.
    Returns a dict of raw results keyed by run name for quick inspection."""
    results = {}
    for curriculum_type in ("none", "complexity"):
        name = f"pilot_sumsquares_{curriculum_type}_s{seed}"
        print(f"\n=== Sumsquares pilot: curriculum={curriculum_type}, seed={seed} ===")
        config = Config(curriculum_type=curriculum_type, task_type="sumsquares", seed=seed,
                         max_steps=max_steps, experiment_name=name)
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

        censored = result.get('test_grok_censored', True)
        print(f"  [pilot] curriculum={curriculum_type}: final_test_acc={result['final_test_acc']:.4f}, "
              f"grok_step={result['test_grok_step']}{' (censored)' if censored else ''}")
        results[name] = result
    return results


def plot_sumsquares_pilot(results: Dict[str, Dict], save_dir: Path = Path("final_results/figures")):
    """Quick train/test curve plot for the pilot runs, same visual language
    as Fig. 1 (Grokking Dynamics) so it can be sanity-checked by eye against
    the existing addition/multiplication figures."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6), squeeze=False)
    for ax, (name, result) in zip(axes[0], results.items()):
        h = result['history']
        ax.plot(h['step'], h['train_acc'], label='Train Accuracy', color='#2E86AB', linewidth=2)
        ax.plot(h['step'], h['test_acc'], label='Test Accuracy', color='#8E44AD', linewidth=2)
        ax.fill_between(h['step'], h['train_acc'], h['test_acc'], alpha=0.2, color='orange', label='Generalization Gap')
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% Threshold')
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'sumsquares_pilot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_dir / 'sumsquares_pilot.png'}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | AMP enabled: {AMP_ENABLED}")

    # # ------------------------------------------------------------------
    # # STEP 0 (recommended first run): confirm sumsquares groks cleanly and
    # # check its intrinsic difficulty/curriculum-dependence before spending
    # # compute on the full transplant suite. Cheap: 2 runs x max_steps.
    # # ------------------------------------------------------------------
    # pilot_results = run_sumsquares_pilot(seed=42, max_steps=30000)
    # plot_sumsquares_pilot(pilot_results)

    # print("\nPilot complete. If sumsquares grokked cleanly (test_acc >= 0.95, not censored,"
    #       " ideally requiring the curriculum the same way addition/multiplication did), proceed with:\n"
    #       "  probe_mult2sq, donor_paths_mult2sq = run_teacher_probe_suite(\n"
    #       "      donor_task='multiplication', recipient_task='sumsquares', seeds=(42,43,44))\n"
    #       "  probe_sq2mult, donor_paths_sq2mult = run_teacher_probe_suite(\n"
    #       "      donor_task='sumsquares', recipient_task='multiplication', seeds=(42,43,44))\n"
    #       "  summary_mult2sq = summarize_teacher_probe(probe_mult2sq, direction_label='mult>sumsquares')\n"
    #       "  summary_sq2mult = summarize_teacher_probe(probe_sq2mult, direction_label='sumsquares>mult')\n"
    #       "  plot_teacher_probe(summary_mult2sq, direction_label='mult>sumsquares')\n"
    #       "  plot_teacher_probe(summary_sq2mult, direction_label='sumsquares>mult')\n"
    #       "  plot_bidirectional_comparison(summary_mult2sq, summary_sq2mult,\n"
    #       "      label_a='mult -> sumsquares', label_b='sumsquares -> mult',\n"
    #       "      filename='bidirectional_transfer_mult_sumsquares.png')\n"
    #       "\n"
    #       "For the component-ablation ('Legos') probe extended to 3 tasks:\n"
    #       "  all_dfs_3way = run_full_ablation_suite(seeds=(42,43,44),\n"
    #       "      mixed_tasks=('addition','multiplication','sumsquares'))\n"
    #       "  summary_3way = summarize_dissociation_across_seeds(all_dfs_3way,\n"
    #       "      tasks=('addition','multiplication','sumsquares'))\n"
    #       "\n"
    #       "MECHANISTIC LINKAGE (Claim A): once you have BOTH\n"
    #       "  (a) the addition/multiplication component-ablation probe run\n"
    #       "      (run_full_ablation_suite with mixed_tasks=('addition','multiplication')),\n"
    #       "  (b) the real-donor standalone embeddings for addition and multiplication\n"
    #       "      (already saved by run_teacher_probe_suite as donor_paths['real'][seed]\n"
    #       "       for whichever donor_task you ran it with -- you need BOTH tasks' real\n"
    #       "       donors at the SAME seed, so run_teacher_probe_suite once with\n"
    #       "       donor_task='addition' and once with donor_task='multiplication', same seeds),\n"
    #       "you can compute the subspace-alignment figure that links the two:\n"
    #       "\n"
    #       "  for seed in (42, 43, 44):\n"
    #       "      run_subspace_alignment_analysis(\n"
    #       "          seed=seed, tasks=('addition','multiplication'),\n"
    #       "          donor_embedding_paths={\n"
    #       "              'addition': donor_paths_add_standalone['real'][seed],\n"
    #       "              'multiplication': donor_paths_mult_standalone['real'][seed],\n"
    #       "          })\n"
    #       "  align_summary = summarize_subspace_alignment_across_seeds(\n"
    #       "      tasks=('addition','multiplication'), seeds=(42,43,44))\n"
    #       "  plot_subspace_alignment_heatmap(\n"
    #       "      align_summary, tasks=('addition','multiplication'),\n"
    #       "      transfer_asymmetry_note=(\n"
    #       "          'Observed: shuffled-donor-finetuned reaches 100% test acc add->mult '\n"
    #       "          'but only ~33-37% mult->add (partial, seed-44 run pending)'))\n"
    #       "\n"
    #       "Read the resulting heatmap's OFF-DIAGONAL cells first: if e.g. the\n"
    #       "multiplication mixed-model subspace aligns much more strongly with the\n"
    #       "addition donor's subspace than the reverse cross-term does, that asymmetry\n"
    #       "is a direct geometric candidate explanation for the observed transfer\n"
    #       "asymmetry -- upgrading 'structure exists and is decomposable' into\n"
    #       "'structure exists, is decomposable, and cross-task alignment predicts\n"
    #       "which transfer direction succeeds.'\n")
    # Standalone real donors for BOTH tasks, same seeds
    _, donor_paths_add_standalone = run_teacher_probe_suite(
    donor_task="addition", recipient_task="multiplication", seeds=(42,43,44))
    _, donor_paths_mult_standalone = run_teacher_probe_suite(
        donor_task="multiplication", recipient_task="addition", seeds=(42,43,44))

    # Ablation probe on the 2-way mixed model (as before)
    all_dfs = run_full_ablation_suite(seeds=(42,43,44),
        mixed_tasks=("addition","multiplication"))

    # NEW: subspace alignment, linking the two
    for seed in (42,43,44):
        run_subspace_alignment_analysis(
            seed=seed, tasks=("addition","multiplication"),
            donor_embedding_paths={
                "addition": donor_paths_add_standalone["real"][seed],
                "multiplication": donor_paths_mult_standalone["real"][seed],
            })
    align_summary = summarize_subspace_alignment_across_seeds(
        tasks=("addition","multiplication"), seeds=(42,43,44))
    plot_subspace_alignment_heatmap(align_summary, tasks=("addition","multiplication"),
        transfer_asymmetry_note="Observed: shuffled-donor-finetuned reaches 100% add->mult but only ~33-37% mult->add")
