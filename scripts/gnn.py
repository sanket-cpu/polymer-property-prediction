"""
GNN for polymer property prediction (plain PyTorch, no PyG).

Architecture : MPNN (message-passing neural net), 4 layers, hidden=128
               Multi-task: shared backbone, separate Tg and Egc heads
               Mean + sum global readout -> 256-dim graph embedding
Input        : monomer SMILES with * attachment points as a special atom type
Validation   : same 5-fold StratifiedGroupKFold as baseline.py (SEED=42)
Output       : outputs/gnn_oof.csv        OOF predictions (for blending)
               outputs/gnn_submission.csv  standalone submission

Run from project root:
    python3 scripts/gnn.py
"""

import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold

# ── Logging ────────────────────────────────────────────────────────────────────

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.makedirs("logs", exist_ok=True)
_log_path = f"logs/gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_log_fh   = open(_log_path, "w", encoding="utf-8", buffering=1)

class _Tee:
    def __init__(self, t, f):
        self.t, self.f = t, f
    def write(self, d): self.t.write(d); self.f.write(d)
    def flush(self):    self.t.flush();  self.f.flush()

sys.stdout = _Tee(sys.stdout, _log_fh)  # type: ignore[assignment]
print(f"Logging to {_log_path}\n")

RDLogger.DisableLog("rdApp.*")

# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
OOF_PATH   = "outputs/gnn_oof.csv"
SUB_PATH   = "outputs/gnn_submission.csv"

SEED        = 42
N_FOLDS     = 5
HIDDEN_DIM  = 128
N_LAYERS    = 4
DROPOUT     = 0.1
LR          = 1e-3
WEIGHT_DECAY= 1e-5
BATCH_SIZE  = 64
MAX_EPOCHS  = 200
PATIENCE    = 25

print(f"HIDDEN_DIM={HIDDEN_DIM}  N_LAYERS={N_LAYERS}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
print(f"BATCH_SIZE={BATCH_SIZE}  LR={LR}  DROPOUT={DROPOUT}\n")

device = torch.device("cpu")

# ── Atom / bond feature encoding ───────────────────────────────────────────────

_ATOM_LIST = ["C","N","O","S","F","Cl","Br","I","Si","P","*"]   # 11 + other = 12
_DEG_LIST  = [0, 1, 2, 3, 4, 5, 6]                              #  7 + other =  8
_CHG_LIST  = [-2, -1, 0, 1, 2]                                  #  5 + other =  6
_HYB_LIST  = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # 6 + other = 7
_BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]  # 4 + other = 5

ATOM_DIM = (len(_ATOM_LIST) + 1) + (len(_DEG_LIST) + 1) + (len(_CHG_LIST) + 1) + (len(_HYB_LIST) + 1) + 2
# 12 + 8 + 6 + 7 + 2 = 35
BOND_DIM = (len(_BOND_LIST) + 1) + 2
# 5 + 2 = 7


def _one_hot(val, choices):
    enc = [0] * (len(choices) + 1)
    enc[choices.index(val) if val in choices else len(choices)] = 1
    return enc


def _atom_feats(atom):
    return (
        _one_hot(atom.GetSymbol(),          _ATOM_LIST)   # 12
        + _one_hot(atom.GetDegree(),        _DEG_LIST)    #  8
        + _one_hot(atom.GetFormalCharge(),  _CHG_LIST)    #  6
        + _one_hot(atom.GetHybridization(), _HYB_LIST)    #  7
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]  # 2
    )  # 35 total


def _bond_feats(bond):
    return (
        _one_hot(bond.GetBondType(), _BOND_LIST)              # 5
        + [int(bond.IsInRing()), int(bond.GetIsConjugated())] # 2
    )  # 7 total


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    node_feats = torch.tensor([_atom_feats(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_feats(bond)
        edges += [[i, j], [j, i]]
        edge_feats += [bf, bf]
    if not edges:             # isolated atom: self-loop so layers can run
        edges = [[0, 0]]
        edge_feats = [[0] * BOND_DIM]
    edge_index = torch.tensor(edges, dtype=torch.long).T       # [2, E]
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float)   # [E, 7]
    return {"node_feats": node_feats, "edge_index": edge_index,
            "edge_attr": edge_attr, "n": node_feats.size(0)}


# ── Datasets ───────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    """Wraps precomputed graphs with normalised targets and original train indices.

    Only keeps rows where the graph was successfully built. The orig_idx
    passed through lets us track which rows made it into OOF output without
    relying on a fragile pointer counter that would drift on parse failures.
    """
    def __init__(self, graphs, targets, ttypes, orig_indices):
        self.items = []
        for g, tgt, tt, oidx in zip(graphs, targets, ttypes, orig_indices):
            if g is not None:
                self.items.append((g, float(tgt), str(tt), int(oidx)))

    def __len__(self):       return len(self.items)
    def __getitem__(self, i): return self.items[i]


class TestGraphDataset(Dataset):
    def __init__(self, df):
        self.items = []
        for _, row in df.iterrows():
            g = mol_to_graph(row["smiles"])
            self.items.append((g, int(row["id"]), str(row["target_type"])))

    def __len__(self):       return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate_train(batch):
    graphs, targets, ttypes, orig_indices = zip(*batch)
    node_list, ei_list, ea_list, bi_list = [], [], [], []
    offset = 0
    for b, g in enumerate(graphs):
        n = g["n"]
        node_list.append(g["node_feats"])
        ei_list.append(g["edge_index"] + offset)
        ea_list.append(g["edge_attr"])
        bi_list.append(torch.full((n,), b, dtype=torch.long))
        offset += n
    return {
        "node_feats":   torch.cat(node_list),
        "edge_index":   torch.cat(ei_list, dim=1),
        "edge_attr":    torch.cat(ea_list),
        "batch_idx":    torch.cat(bi_list),
        "batch_size":   len(batch),
        "targets":      torch.tensor(targets, dtype=torch.float),
        "ttypes":       list(ttypes),
        "orig_indices": list(orig_indices),
    }


def _collate_test(batch):
    graphs, ids, ttypes = zip(*batch)
    valid = [(g, rid, tt) for g, rid, tt in zip(graphs, ids, ttypes) if g is not None]
    n_fail = len(batch) - len(valid)
    if n_fail:
        print(f"    [warn] {n_fail} test graphs failed, fallback to global mean")
    if not valid:
        return {"node_feats": None, "ids": list(ids), "ttypes": list(ttypes)}
    vg, vids, vtts = zip(*valid)
    node_list, ei_list, ea_list, bi_list = [], [], [], []
    offset = 0
    for b, g in enumerate(vg):
        n = g["n"]
        node_list.append(g["node_feats"])
        ei_list.append(g["edge_index"] + offset)
        ea_list.append(g["edge_attr"])
        bi_list.append(torch.full((n,), b, dtype=torch.long))
        offset += n
    return {
        "node_feats": torch.cat(node_list),
        "edge_index": torch.cat(ei_list, dim=1),
        "edge_attr":  torch.cat(ea_list),
        "batch_idx":  torch.cat(bi_list),
        "batch_size": len(vg),
        "ids":        list(vids),
        "ttypes":     list(vtts),
    }


# ── Scatter helpers ────────────────────────────────────────────────────────────

def _scatter_mean(src, idx, size):
    out   = src.new_zeros(size, src.size(-1))
    count = src.new_zeros(size, 1)
    out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(0, idx.unsqueeze(1), src.new_ones(len(idx), 1))
    return out / count.clamp(min=1)


def _scatter_sum(src, idx, size):
    out = src.new_zeros(size, src.size(-1))
    out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
    return out


# ── Model ──────────────────────────────────────────────────────────────────────

class _MPNNLayer(nn.Module):
    def __init__(self, node_dim, bond_dim):
        super().__init__()
        # Message: concatenate sender node features + bond features, project to node_dim
        self.msg  = nn.Sequential(
            nn.Linear(node_dim + bond_dim, node_dim), nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        # Update: GRU treats aggregated messages as input, current h as hidden state
        self.gru  = nn.GRUCell(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m   = self.msg(torch.cat([h[src], edge_attr], dim=-1))  # [E, node_dim]
        agg = _scatter_mean(m, dst, h.size(0))                  # [N, node_dim]
        return self.norm(self.gru(agg, h))


class MPNN(nn.Module):
    def __init__(self, atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
                 hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embed  = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList([_MPNNLayer(hidden, bond_dim) for _ in range(n_layers)])
        self.drop   = nn.Dropout(dropout)

        readout = hidden * 2   # mean-pool + sum-pool concatenated -> 256
        self.tg_head  = nn.Sequential(
            nn.Linear(readout, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.egc_head = nn.Sequential(
            nn.Linear(readout, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, node_feats, edge_index, edge_attr, batch_idx, batch_size):
        h = self.embed(node_feats)
        for layer in self.layers:
            h = self.drop(layer(h, edge_index, edge_attr))
        g = torch.cat([
            _scatter_mean(h, batch_idx, batch_size),
            _scatter_sum(h,  batch_idx, batch_size),
        ], dim=-1)                                  # [B, 256]
        return self.tg_head(g).squeeze(-1), self.egc_head(g).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(tr_ds, val_ds, tg_mean, tg_std, egc_mean, egc_std, fold_seed):
    gen = torch.Generator()
    gen.manual_seed(fold_seed)
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=_collate_train, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=_collate_train)

    torch.manual_seed(fold_seed)
    model = MPNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_r2, best_state, wait = -np.inf, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        for b in tr_dl:
            tg_p, egc_p = model(b["node_feats"], b["edge_index"], b["edge_attr"],
                                b["batch_idx"], b["batch_size"])
            tg_idx  = [i for i, t in enumerate(b["ttypes"]) if t == "tg"]
            egc_idx = [i for i, t in enumerate(b["ttypes"]) if t == "egc"]
            loss = torch.tensor(0.0)
            if tg_idx:
                loss = loss + F.mse_loss(tg_p[tg_idx],  b["targets"][tg_idx])
            if egc_idx:
                loss = loss + F.mse_loss(egc_p[egc_idx], b["targets"][egc_idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # ── Validate ──
        model.eval()
        tg_true, tg_hat, egc_true, egc_hat = [], [], [], []
        with torch.no_grad():
            for b in val_dl:
                tg_p, egc_p = model(b["node_feats"], b["edge_index"], b["edge_attr"],
                                    b["batch_idx"], b["batch_size"])
                for i, tt in enumerate(b["ttypes"]):
                    raw = b["targets"][i].item()
                    if tt == "tg":
                        tg_true.append(raw * tg_std  + tg_mean)
                        tg_hat.append(tg_p[i].item()  * tg_std  + tg_mean)
                    else:
                        egc_true.append(raw * egc_std + egc_mean)
                        egc_hat.append(egc_p[i].item() * egc_std + egc_mean)

        r2_tg  = r2_score(tg_true,  tg_hat)  if len(tg_true)  > 1 else 0.0
        r2_egc = r2_score(egc_true, egc_hat) if len(egc_true) > 1 else 0.0
        r2     = (r2_tg + r2_egc) / 2

        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep {epoch:3d}  Tg={r2_tg:+.4f}  Egc={r2_egc:+.4f}  mean={r2:+.4f}")

        if r2 > best_r2:
            best_r2   = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop ep={epoch}  best mean R²={best_r2:+.4f}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r2


# ── Main ───────────────────────────────────────────────────────────────────────

start = datetime.now()
print(f"Started {start.strftime('%H:%M:%S')}  |  ATOM_DIM={ATOM_DIM}  BOND_DIM={BOND_DIM}\n")

# ── Load + preprocess (same dedup logic as baseline.py) ───────────────────────

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  raw train: {len(train):,}  test: {len(test):,}")


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


train["canon_smiles"] = train["smiles"].apply(_canon)
n_raw = len(train)
train = (
    train.groupby(["canon_smiles", "target_type"], as_index=False)
    .agg(smiles=("smiles", "first"), target=("target", "mean"))
)
print(f"  After dedup merge: {n_raw:,} -> {len(train):,}")
train = train.reset_index(drop=True)

smi_all = train["smiles"].values
y_all   = train["target"].values.astype(float)
tt_all  = train["target_type"].values
groups  = train["canon_smiles"].values

# ── Target normalisation (global) ─────────────────────────────────────────────

tg_mask  = tt_all == "tg"
egc_mask = tt_all == "egc"
tg_mean,  tg_std  = y_all[tg_mask].mean(),  y_all[tg_mask].std()
egc_mean, egc_std = y_all[egc_mask].mean(), y_all[egc_mask].std()
print(f"\n  Tg  : mean={tg_mean:.1f}  std={tg_std:.1f}  n={tg_mask.sum()}")
print(f"  Egc : mean={egc_mean:.3f}  std={egc_std:.3f}  n={egc_mask.sum()}")

y_norm = y_all.copy()
y_norm[tg_mask]  = (y_all[tg_mask]  - tg_mean)  / (tg_std  + 1e-8)
y_norm[egc_mask] = (y_all[egc_mask] - egc_mean) / (egc_std + 1e-8)

# ── Build all train graphs once ───────────────────────────────────────────────

print("\nBuilding train graphs (once)...")
all_graphs = [mol_to_graph(s) for s in smi_all]
n_fail = sum(1 for g in all_graphs if g is None)
print(f"  Build failures: {n_fail}/{len(smi_all)}")

# ── Build test graphs ─────────────────────────────────────────────────────────

print("Building test graphs...")
test_ds = TestGraphDataset(test)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_test)
print(f"  Test molecules: {len(test_ds)}")

# Per-molecule accumulators for fold-averaged test predictions
test_tg_acc  = {int(row["id"]): 0.0 for _, row in test.iterrows()}
test_egc_acc = {int(row["id"]): 0.0 for _, row in test.iterrows()}
test_cnt     = {int(row["id"]): 0   for _, row in test.iterrows()}

# ── 5-fold CV ─────────────────────────────────────────────────────────────────

sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(sgkf.split(smi_all, tt_all, groups))

oof_rows = []
fold_r2s = []

print(f"\n{'='*60}")
print(f"  5-FOLD CV  (SEED={SEED})")
print(f"{'='*60}")

for fold, (tr_idx, val_idx) in enumerate(splits, 1):
    print(f"\n── Fold {fold}/{N_FOLDS} ──")

    tr_ds  = GraphDataset(
        [all_graphs[i] for i in tr_idx],
        y_norm[tr_idx], tt_all[tr_idx], tr_idx,
    )
    val_ds = GraphDataset(
        [all_graphs[i] for i in val_idx],
        y_norm[val_idx], tt_all[val_idx], val_idx,
    )
    print(f"  train={len(tr_ds)}  val={len(val_ds)}")

    model, fold_r2 = train_fold(
        tr_ds, val_ds, tg_mean, tg_std, egc_mean, egc_std, fold_seed=SEED + fold,
    )
    fold_r2s.append(fold_r2)
    print(f"  Fold {fold} best mean R² = {fold_r2:+.4f}")

    # ── OOF predictions ──
    model.eval()
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_train)
    with torch.no_grad():
        for b in val_dl:
            tg_p, egc_p = model(b["node_feats"], b["edge_index"], b["edge_attr"],
                                b["batch_idx"], b["batch_size"])
            for i, (tt, oidx) in enumerate(zip(b["ttypes"], b["orig_indices"])):
                raw = b["targets"][i].item()
                if tt == "tg":
                    y_true = raw * tg_std  + tg_mean
                    y_pred = tg_p[i].item()  * tg_std  + tg_mean
                else:
                    y_true = raw * egc_std + egc_mean
                    y_pred = egc_p[i].item() * egc_std + egc_mean
                oof_rows.append({
                    "train_idx":   oidx,
                    "canon_smiles": groups[oidx],
                    "target_type": tt,
                    "y_true":      y_true,
                    "y_pred":      y_pred,
                })

    # ── Test predictions (accumulate, divide by N_FOLDS at the end) ──
    with torch.no_grad():
        for b in test_dl:
            if b["node_feats"] is None:
                continue
            tg_p, egc_p = model(b["node_feats"], b["edge_index"], b["edge_attr"],
                                b["batch_idx"], b["batch_size"])
            for j, (rid, _tt) in enumerate(zip(b["ids"], b["ttypes"])):
                test_tg_acc[rid]  += tg_p[j].item()  * tg_std  + tg_mean
                test_egc_acc[rid] += egc_p[j].item() * egc_std + egc_mean
                test_cnt[rid]     += 1

# ── OOF scoring ───────────────────────────────────────────────────────────────

oof = pd.DataFrame(oof_rows)
oof_tg  = oof[oof["target_type"] == "tg"]
oof_egc = oof[oof["target_type"] == "egc"]
r2_tg  = r2_score(oof_tg["y_true"],  oof_tg["y_pred"])
r2_egc = r2_score(oof_egc["y_true"], oof_egc["y_pred"])
cv     = (r2_tg + r2_egc) / 2

print(f"\n{'='*60}")
print(f"  OOF RESULTS")
print(f"{'='*60}")
for i, r in enumerate(fold_r2s, 1):
    print(f"  Fold {i}: best val mean R² = {r:+.4f}")
print(f"\n  OOF R²(Tg)  = {r2_tg:+.4f}")
print(f"  OOF R²(Egc) = {r2_egc:+.4f}")
print(f"  Fold R² std = {np.std(fold_r2s):.4f}")
print(f"\n  >>> GNN CV score = {cv:+.4f} <<<")

# ── Save OOF predictions ──────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)
oof.to_csv(OOF_PATH, index=False)
print(f"\nOOF saved -> {OOF_PATH}  ({len(oof):,} rows)")

# ── Build submission ──────────────────────────────────────────────────────────

rows = []
for _, row in test.iterrows():
    rid = int(row["id"])
    cnt = test_cnt[rid]
    if row["target_type"] == "tg":
        pred = test_tg_acc[rid]  / cnt if cnt > 0 else tg_mean
    else:
        pred = test_egc_acc[rid] / cnt if cnt > 0 else egc_mean
    rows.append({"id": rid, "target": pred})

submission = pd.DataFrame(rows).sort_values("id")
submission.to_csv(SUB_PATH, index=False)
print(f"Submission saved -> {SUB_PATH}  ({len(submission):,} rows)\n")
print(submission.head(5).to_string(index=False))

elapsed = datetime.now() - start
print(f"\nDone. Total time: {elapsed}")
