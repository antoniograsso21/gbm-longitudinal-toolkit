# debug_mi.py
#  PYTHONPATH=. uv run python scripts/debug_mi.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

df = pd.read_parquet("data/processed/preprocessing/dataset_engineered.parquet")

exclude = {"Patient","Timepoint","target","target_encoded","is_baseline_scan","is_nadir_scan"}
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols].values
y = df["target_encoded"].values

# sklearn MI come proxy veloce — non Kraskov ma sufficiente per diagnostica
mi_scores = mutual_info_classif(X, y, random_state=42)
top20_idx = np.argsort(mi_scores)[::-1][:20]

print("Top 20 features by MI score:")
for i in top20_idx:
    print(f"  {mi_scores[i]:.4f}  {feature_cols[i]}")

print(f"\nFeatures with MI > 0.05: {(mi_scores > 0.05).sum()}")
print(f"Features with MI > 0.01: {(mi_scores > 0.01).sum()}")
print(f"Features with MI > 0.001: {(mi_scores > 0.001).sum()}")