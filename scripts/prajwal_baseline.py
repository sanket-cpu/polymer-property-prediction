# ==================== AGGRESSIVE CREATIVE STACKING v3 ====================

!pip install -q rdkit xgboost lightgbm

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

def get_aggressive_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1600, dtype=np.float32)
    
    feats = []
    
    # Multiple Morgan radii
    for radius in [2, 3]:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
        arr = np.zeros(1024, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        feats.append(arr)
    
    # Rich descriptors
    try:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList[:220]])
        desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        feats.append(desc)
    except:
        feats.append(np.zeros(220, dtype=np.float32))
    
    # Advanced polymer features
    custom = np.array([
        len(smiles),
        smiles.count('(') + smiles.count(')'),
        Chem.rdMolDescriptors.CalcNumRings(mol),
        Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
        Chem.Descriptors.MolWt(mol),
        Chem.Descriptors.MolLogP(mol),
        Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
        smiles.count('O') + smiles.count('N') + smiles.count('S') + smiles.count('F') + smiles.count('Cl'),
        smiles.count('C(=O)'), smiles.count('OC(=O)'), smiles.count('NC(=O)'),
        smiles.count('c'), smiles.count('['),
        # Flexibility / polarity proxies
        smiles.count('CC') / (len(smiles)+1),
        smiles.count('O') / (len(smiles)+1),
    ], dtype=np.float32)
    feats.append(custom)
    
    features = np.concatenate(feats)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e7, 1e7)
    return features.astype(np.float32)

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Extracting aggressive features...")
tg_mask = train['target_type'] == 'tg'
X_tg = np.array([get_aggressive_features(s) for s in train[tg_mask]['smiles']])
y_tg = train[tg_mask]['target'].values

egc_mask = train['target_type'] == 'egc'
X_egc = np.array([get_aggressive_features(s) for s in train[egc_mask]['smiles']])
y_egc = train[egc_mask]['target'].values

# Scale
scaler_tg = RobustScaler()
X_tg_s = scaler_tg.fit_transform(X_tg)
X_tg_s = np.nan_to_num(X_tg_s, 0)

scaler_egc = RobustScaler()
X_egc_s = scaler_egc.fit_transform(X_egc)
X_egc_s = np.nan_to_num(X_egc_s, 0)

def train_aggressive_stacking(X, y, name=""):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Level 0 - Base models
    bases = []
    params_list = [
        {'n_estimators': 900, 'lr': 0.028, 'depth': 9},
        {'n_estimators': 650, 'lr': 0.045, 'depth': 7},
        {'n_estimators': 750, 'lr': 0.032, 'depth': 8}
    ]
    
    for i, p in enumerate(params_list):
        if i < 2:
            m = xgb.XGBRegressor(n_estimators=p['n_estimators'], learning_rate=p['lr'], 
                                 max_depth=p['depth'], subsample=0.83, colsample_bytree=0.77, 
                                 random_state=42+i, n_jobs=-1)
        else:
            m = lgb.LGBMRegressor(n_estimators=p['n_estimators'], learning_rate=p['lr'], 
                                  max_depth=p['depth'], subsample=0.83, colsample_bytree=0.77, 
                                  random_state=42+i, verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train)
        bases.append(m)
    
    # Level 1 - Blend
    level1_train = np.column_stack([m.predict(X_train) for m in bases])
    level1_val = np.column_stack([m.predict(X_val) for m in bases])
    
    # Level 2 - Meta
    meta = xgb.XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=5, random_state=99)
    meta.fit(level1_train, y_train)
    
    final = meta.predict(level1_val)
    print(f"{name} Aggressive Stacking R²: {r2_score(y_val, final):.4f}")
    return bases, meta

print("\n=== Training Tg ===")
tg_bases, tg_meta = train_aggressive_stacking(X_tg_s, y_tg, "Tg")

print("\n=== Training Egc ===")
egc_bases, egc_meta = train_aggressive_stacking(X_egc_s, y_egc, "Egc")

# Submission
print("\nGenerating submission...")
preds = []
for _, row in test.iterrows():
    feat = get_aggressive_features(row['smiles'])
    if row['target_type'] == 'tg':
        feat_s = scaler_tg.transform(feat.reshape(1,-1))
        feat_s = np.nan_to_num(feat_s, 0)
        l1 = np.column_stack([m.predict(feat_s) for m in tg_bases])
        pred = tg_meta.predict(l1)[0]
    else:
        feat_s = scaler_egc.transform(feat.reshape(1,-1))
        feat_s = np.nan_to_num(feat_s, 0)
        l1 = np.column_stack([m.predict(feat_s) for m in egc_bases])
        pred = egc_meta.predict(l1)[0]
    preds.append({'id': int(row['id']), 'target': round(float(pred), 4)})

sub = pd.DataFrame(preds)
sub.to_csv('submission.csv', index=False)
print("✅ Submission saved as submission.csv")
print(sub.head())
