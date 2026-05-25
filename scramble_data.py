"""
scramble_data.py — SWAP + JITTER version
Randomly reassigns binding data between systems, then adds small jitter.
Distribution is preserved. Per-compound assignments are destroyed.
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

REAL_DATA  = Path("/home/jwallace-iit.local/projects/PET_Contaminant/data/_50_")
OUT_DATA   = Path("data/_50_")
SEED       = 42
JITTER_PCT = 0.03   # 3% std jitter — tweak if needed
rng        = np.random.default_rng(SEED)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
sys_df = pd.read_csv(REAL_DATA / "summaries/summary_systems.csv")
ana_df = pd.read_csv(REAL_DATA / "summaries/summary_analytes.csv")
atm_df = pd.read_parquet(REAL_DATA / "summaries/summary_atoms.parquet")

SYSTEM_COL = "system"

# ── 2. SWAP HELPER ────────────────────────────────────────────────────────────
def swap_and_jitter(df, system_col=SYSTEM_COL, jitter_pct=JITTER_PCT):
    """
    - Keep system names in place
    - Generate a random permutation of row indices
    - For every numeric column, pull values from the permuted rows
    - Add tiny Gaussian jitter (% of column std) to break exact matches
    """
    df = df.copy()
    n  = len(df)

    perm = rng.permutation(n)

    for col in df.select_dtypes(include=[np.number]).columns:
        original_vals = df[col].values.copy()
        swapped       = original_vals[perm]
        noise         = rng.normal(0, np.std(swapped) * jitter_pct, size=n)
        df[col]       = (swapped + noise).clip(original_vals.min(), original_vals.max())

    return df

# ── 3. WRITE SUMMARIES ────────────────────────────────────────────────────────
out_sum = OUT_DATA / "summaries"
out_sum.mkdir(parents=True, exist_ok=True)

swap_and_jitter(sys_df).to_csv(    out_sum / "summary_systems.csv",   index=False)
swap_and_jitter(ana_df).to_csv(    out_sum / "summary_analytes.csv",  index=False)
swap_and_jitter(atm_df).to_parquet(out_sum / "summary_atoms.parquet", index=False)

print("Summaries written.")

# ── 4. MOL2 — keep real structures under real names ───────────────────────────
real_systems = sys_df[SYSTEM_COL].unique()
copied = 0
for name in real_systems:
    src = REAL_DATA / name / "LIG.mol2"
    dst = OUT_DATA / name / "LIG.mol2"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

print(f"Copied {copied} LIG.mol2 files under real system names.")
print("Done.")
