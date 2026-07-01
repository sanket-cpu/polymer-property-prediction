import os
import sys
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

# ── Environment & Config ──────────────────────────────────────────────────────
# sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
SEED = 93  # Preserving your baseline seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

EARLY_STOPPING_ROUNDS = 50
N_TRIALS = 5  # Sightly reduced from 50 to optimize runtime given two full tuning phases
TUNE_N_ESTIMATORS = 150
FINAL_N_ESTIMATORS = 300

LGB_FIXED = {
    "n_estimators": FINAL_N_ESTIMATORS,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ── 1. Tabular Feature Extraction ─────────────────────────────────────────────
def _desc_one(smi):
    return Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smi))


def _fp_one(smi):
    mol = Chem.MolFromSmiles(smi)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))


def _maccs_one(smi):
    mol = Chem.MolFromSmiles(smi)
    return list(MACCSkeys.GenMACCSKeys(mol))


def _topo_one(smi):
    mol = Chem.MolFromSmiles(smi)
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    dmat = Chem.GetDistanceMatrix(mol)
    star_dist = dmat[star_idx[0], star_idx[1]]
    diameter = dmat.max()
    return {
        "star_distance": star_dist,
        "star_distance_frac": star_dist / diameter if diameter > 0 else 0.0,
    }


def compute_features(df):
    smiles = df["smiles"].tolist()
    descs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_desc_one)(s) for s in smiles
    )
    fps = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fp_one)(s) for s in smiles
    )
    maccs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_maccs_one)(s) for s in smiles
    )
    topo = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_topo_one)(s) for s in smiles
    )

    desc_df = pd.DataFrame(descs, index=df.index)
    fp_df = pd.DataFrame(
        fps, index=df.index, columns=[f"morgan_{i}" for i in range(2048)]
    )
    maccs_df = pd.DataFrame(
        maccs, index=df.index, columns=[f"maccs_{i}" for i in range(167)]
    )
    topo_df = pd.DataFrame(topo, index=df.index)

    combined = pd.concat([desc_df, fp_df, maccs_df, topo_df], axis=1)
    return combined.replace([np.inf, -np.inf], np.nan)


# ── 2. Graph Engineering (Macrocycle Proxy) ───────────────────────────────────
ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "I",
    "*",
    "other",
]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    "other",
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _one_hot(val, choices):
    vec = [0.0] * (len(choices) + 1)
    idx = choices.index(val) if val in choices else len(choices)
    vec[idx] = 1.0
    return vec


def _atom_features(atom, is_attachment):
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_SYMBOLS[:-1])
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS[:-1])
    feats.append(atom.GetDegree() / 4.0)
    feats.append(atom.GetFormalCharge() / 2.0)
    feats.append(atom.GetTotalNumHs() / 4.0)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(1.0 if is_attachment else 0.0)
    return feats


def _bond_features(bond):
    feats = _one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]

    mol = Chem.RWMol(mol)
    if len(star_idx) == 2:
        if mol.GetBondBetweenAtoms(star_idx[0], star_idx[1]) is None:
            mol.AddBond(star_idx[0], star_idx[1], Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol, catchErrors=True)

    attach_set = set(star_idx)
    x = torch.tensor(
        [_atom_features(a, a.GetIdx() in attach_set) for a in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * (len(BOND_TYPES) + 2)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


NODE_DIM = len(_atom_features(Chem.MolFromSmiles("CC").GetAtomWithIdx(0), False))
EDGE_DIM = 7 # Corrected from len(BOND_TYPES) + 2


# ── 3. Multi-Task Deep GNN Encoder ────────────────────────────────────────────
class PolymerGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=4, dropout=0.15):
        super().__init__()
        self.lin_in = nn.Linear(node_dim, hidden)
        self.edge_encoders = nn.ModuleList(
            [nn.Linear(edge_dim, hidden) for _ in range(n_layers)]
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

        embed_dim = hidden * 2  # Concat mean + max pooling
        self.head_tg = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_egc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.embed_dim = embed_dim

    def encode(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        h = self.lin_in(x)
        for conv, norm, edge_enc in zip(
            self.convs, self.norms, self.edge_encoders
        ):
            e = edge_enc(edge_attr)
            h = conv(h, edge_index, e)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        return torch.cat([g_mean, g_max], dim=1)

    def forward(self, data):
        emb = self.encode(data)
        return (
            self.head_tg(emb).squeeze(-1),
            self.head_egc(emb).squeeze(-1),
            emb,
        )


def train_gnn_fold(
    train_graphs,
    train_y,
    train_type,
    val_graphs,
    val_y,
    val_type,
    epochs=200,
    patience=25,
    lr=1e-3,
    batch_size=64,
):
    model = PolymerGNN(NODE_DIM, EDGE_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8
    )

    def batches(graphs, y, ttype, shuffle):
        idx = np.arange(len(graphs))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            sel = idx[i : i + batch_size]
            g = Batch.from_data_list([graphs[j] for j in sel]).to(DEVICE)
            yield g, y[sel], ttype[sel]

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for g, yb, tb in batches(
            train_graphs, train_y, train_type, shuffle=True
        ):
            opt.zero_grad()
            pred_tg, pred_egc, _ = model(g)
            mask_tg = torch.tensor(tb == "tg", device=DEVICE)
            mask_egc = torch.tensor(tb == "egc", device=DEVICE)
            yb_t = torch.tensor(yb, dtype=torch.float, device=DEVICE)
            loss = 0.0
            if mask_tg.any():
                loss = loss + F.smooth_l1_loss(pred_tg[mask_tg], yb_t[mask_tg])
            if mask_egc.any():
                loss = loss + F.smooth_l1_loss(
                    pred_egc[mask_egc], yb_t[mask_egc]
                )
            loss.backward()
            opt.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for g, yb, tb in batches(
                val_graphs, val_y, val_type, shuffle=False
            ):
                pred_tg, pred_egc, _ = model(g)
                yb_t = torch.tensor(yb, dtype=torch.float, device=DEVICE)
                mask_tg = torch.tensor(tb == "tg", device=DEVICE)
                mask_egc = torch.tensor(tb == "egc", device=DEVICE)
                if mask_tg.any():
                    val_loss += F.smooth_l1_loss(
                        pred_tg[mask_tg], yb_t[mask_tg], reduction="sum"
                    ).item()
                if mask_egc.any():
                    val_loss += F.smooth_l1_loss(
                        pred_egc[mask_egc], yb_t[mask_egc], reduction="sum"
                    ).item()
                n += len(yb)
        val_loss /= max(n, 1)
        sched.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = (
                val_loss,
                {k: v.clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_gnn(model, graphs, ttypes, batch_size=128):
    model.eval()
    preds, embs = [], []
    for i in range(0, len(graphs), batch_size):
        g = Batch.from_data_list(graphs[i : i + batch_size]).to(DEVICE)
        pred_tg, pred_egc, emb = model(g)
        tb = ttypes[i : i + batch_size]
        out = np.where(
            tb == "tg", pred_tg.cpu().numpy(), pred_egc.cpu().numpy()
        )
        preds.append(out)
        embs.append(emb.cpu().numpy())
    return np.concatenate(preds), np.concatenate(embs)


# ── 4. Main Shared Optimization & Cross-Validation Harness ───────────────────
def tune_lgbm(X, y, cv_splits, strat_label, ttype):
    def objective(trial):
        params = {
            **LGB_FIXED,
            "n_estimators": TUNE_N_ESTIMATORS,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
        }
        scores = []
        for tr_idx, val_idx in cv_splits:
            mask_tr = strat_label[tr_idx] == ttype
            mask_val = strat_label[val_idx] == ttype
            X_tr, X_val = X.iloc[tr_idx][mask_tr], X.iloc[val_idx][mask_val]
            y_tr, y_val = y[tr_idx][mask_tr], y[val_idx][mask_val]

            model = LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
                ],
            )
            scores.append(r2_score(y_val, model.predict(X_val)))
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    return {**LGB_FIXED, **study.best_params}


# ── Execution Pipeline ────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = datetime.now()
    print(f"Pipeline started at {start.strftime('%H:%M:%S')}\n")

    # Load data
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y_train = train["target"].values
    strat_label = train["target_type"].values.astype(str)
    groups = train["smiles"].values

    test_types = test["target_type"].values.astype(str)

    # Cross-validation protocol setup
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_splits = list(sgkf.split(np.zeros(len(train)), strat_label, groups))

    # Compute structures
    print("Computing hand-crafted tabular features...")
    X_train = compute_features(train)
    X_test = compute_features(test)

    print("Converting RDKit molecular graphs for deep-learning...")
    train_graphs = [smiles_to_graph(s) for s in train["smiles"]]
    test_graphs = [smiles_to_graph(s) for s in test["smiles"]]

    # ── Phase A: LightGBM Baseline Model (with OOF Tracking) ──────────────────
    print(f"\n{'='*58}\n  STAGE 1: LIGHTGBM BASELINE TUNING & CV\n{'='*58}")
    lgbm_baseline_params = {}
    for ttype in ["tg", "egc"]:
        lgbm_baseline_params[ttype] = tune_lgbm(
            X_train, y_train, cv_splits, strat_label, ttype
        )

    oof_lgbm_pred = np.zeros(len(train))
    test_lgbm_pred_sum = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
        for ttype in ["tg", "egc"]:
            mask_tr = strat_label[tr_idx] == ttype
            mask_val = strat_label[val_idx] == ttype

            X_tr, X_val = (
                X_train.iloc[tr_idx][mask_tr],
                X_train.iloc[val_idx][mask_val],
            )
            y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]

            model = LGBMRegressor(**lgbm_baseline_params[ttype])
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
                ],
            )

            oof_lgbm_pred[val_idx[mask_val]] = model.predict(X_val)

            test_mask = test_types == ttype
            if test_mask.any():
                test_lgbm_pred_sum[test_mask] += model.predict(
                    X_test[test_mask]
                )

    test_lgbm_pred = test_lgbm_pred_sum / N_FOLDS

    # ── Phase B: GNN Multi-task Training ──────────────────────────────────────
    print(f"\n{'='*58}\n  STAGE 2: CO-TRAINING GNN BACKBONE\n{'='*58}")
    oof_gnn_pred = np.zeros(len(train))
    oof_gnn_emb = np.zeros((len(train), PolymerGNN(NODE_DIM, EDGE_DIM).embed_dim))
    test_gnn_pred_sum = np.zeros(len(test))
    test_gnn_emb_sum = np.zeros((len(test), oof_gnn_emb.shape[1]))

    for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
        print(f"Training GNN Structural Encoder - Fold {fold}/{N_FOLDS}...")
        gnn_model = train_gnn_fold(
            [train_graphs[i] for i in tr_idx],
            y_train[tr_idx],
            strat_label[tr_idx],
            [train_graphs[i] for i in val_idx],
            y_train[val_idx],
            strat_label[val_idx],
        )

        val_pred, val_emb = predict_gnn(
            gnn_model, [train_graphs[i] for i in val_idx], strat_label[val_idx]
        )
        oof_gnn_pred[val_idx] = val_pred
        oof_gnn_emb[val_idx] = val_emb

        test_pred, test_emb = predict_gnn(gnn_model, test_graphs, test_types)
        test_gnn_pred_sum += test_pred
        test_gnn_emb_sum += test_emb

    test_gnn_pred = test_gnn_pred_sum / N_FOLDS
    test_gnn_emb = test_gnn_emb_sum / N_FOLDS

    # Save Candidate 1: GNN Only
    sub_gnn = test[["id", "target_type"]].copy()
    sub_gnn["target"] = test_gnn_pred
    sub_gnn[["id", "target"]].sort_values("id").to_csv(
        f"{OUT_DIR}/submission_gnn_only.csv", index=False
    )

    # ── Phase C: Fusion A - Out-of-Fold Ridge Stacking Blend ──────────────────
    print(f"\n{'='*58}\n  STAGE 3: META-STACKING VIA RIDGE REGRESSION\n{'='*58}")
    test_stacked_pred = np.zeros(len(test))

    for ttype in ["tg", "egc"]:
        m = strat_label == ttype
        blender = RidgeCV(alphas=np.logspace(-3, 3, 25))
        blender.fit(np.column_stack([oof_gnn_pred[m], oof_lgbm_pred[m]]), y_train[m])

        print(
            f"  {ttype.upper()} Meta-Weights -> GNN: {blender.coef_[0]:.4f} | LightGBM: {blender.coef_[1]:.4f} | Intercept: {blender.intercept_:.4f}"
        )

        test_mask = test_types == ttype
        if test_mask.any():
            test_stacked_pred[test_mask] = blender.predict(
                np.column_stack(
                    [test_gnn_pred[test_mask], test_lgbm_pred[test_mask]]
                )
            )

    sub_stacked = test[["id", "target_type"]].copy()
    sub_stacked["target"] = test_stacked_pred
    sub_stacked[["id", "target"]].sort_values("id").to_csv(
        f"{OUT_DIR}/submission_stacked_blend.csv", index=False
    )

    # ── Phase D: Fusion B - GNN Embedding Augmented LightGBM Retraining ───────
    print(
        f"\n{'='*58}\n  STAGE 4: EMBEDDING RETRAINING VIA EXTENDED ARRAYS\n{'='*58}"
    )

    # Build matrix append configurations
    emb_cols = [f"gnn_emb_{i}" for i in range(oof_gnn_emb.shape[1])]
    X_train_fused = pd.concat(
        [
            X_train.reset_index(drop=True),
            pd.DataFrame(oof_gnn_emb, columns=emb_cols),
        ],
        axis=1,
    )
    X_test_fused = pd.concat(
        [
            X_test.reset_index(drop=True),
            pd.DataFrame(test_gnn_emb, columns=emb_cols),
        ],
        axis=1,
    )

    fused_lgbm_params = {}
    for ttype in ["tg", "egc"]:
        print(f"Optimizing Fused Hyperparameters for {ttype.upper()}...")
        fused_lgbm_params[ttype] = tune_lgbm(
            X_train_fused, y_train, cv_splits, strat_label, ttype
        )

    test_fused_pred_sum = np.zeros(len(test))
    oof_fused_pred = np.zeros(len(train))

    for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
        for ttype in ["tg", "egc"]:
            mask_tr = strat_label[tr_idx] == ttype
            mask_val = strat_label[val_idx] == ttype

            X_tr, X_val = (
                X_train_fused.iloc[tr_idx][mask_tr],
                X_train_fused.iloc[val_idx][mask_val],
            )
            y_tr, y_val = y_train[tr_idx][mask_tr], y_train[val_idx][mask_val]

            model = LGBMRegressor(**fused_lgbm_params[ttype])
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
                ],
            )

            oof_fused_pred[val_idx[mask_val]] = model.predict(X_val)

            test_mask = test_types == ttype
            if test_mask.any():
                test_fused_pred_sum[test_mask] += model.predict(
                    X_test_fused[test_mask]
                )

    test_fused_pred = test_fused_pred_sum / N_FOLDS

    sub_fused = test[["id", "target_type"]].copy()
    sub_fused["target"] = test_fused_pred
    sub_fused[["id", "target"]].sort_values("id").to_csv(
        f"{OUT_DIR}/submission_embedding_fused.csv", index=False
    )

    # ── Final Metric Reports ──────────────────────────────────────────────────
    print(f"\n{'='*58}\n  FINAL VALIDATION METRICS COMPARISON (Mean R²)\n{'='*58}")

    def report_metrics(oof_array, pipeline_name):
        for ttype in ["tg", "egc"]:
            m = strat_label == ttype
            score = r2_score(y_train[m], oof_array[m])
            print(f"  [{pipeline_name}] R²({ttype.upper()}): {score:+.4f}")

    report_metrics(oof_lgbm_pred, "Baseline LightGBM")
    report_metrics(oof_gnn_pred, "Pure GNN Model   ")
    report_metrics(oof_fused_pred, "Embedding Fused  ")

    print(f"\nExecution Complete. Total Time: {datetime.now() - start}")
