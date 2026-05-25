#!/usr/bin/env python3
"""
Generate fake-but-realistic showcase data for the Streamlit dashboard.

Run once from the repository root. The script reads the real local `_50_`
summary data, anonymizes system identifiers, perturbs metrics, and copies
real LIG.mol2 structures into matching fake system folders.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


real_data_dir = os.environ.get("REAL_DATA_DIR")
if not real_data_dir:
    raise SystemExit("Set REAL_DATA_DIR to the real `_50_` data directory before running.")

REAL_DATA = Path(real_data_dir).expanduser()
OUT_DATA = Path("data/_50_")
SEED = 42
SYSTEM_COL = "system"

rng = np.random.default_rng(SEED)


COUNT_COLS = {
    "analyte",
    "n_replicas",
    "frames_min",
    "frames_max",
    "analytes_used",
    "analytes_total",
}

FRACTION_COLS = {
    "binding_frac_mean",
    "binding_frac_ci_lo",
    "binding_frac_ci_hi",
    "f_mean",
    "f_rep1",
    "f_rep2",
    "f_rep3",
    "p_bound",
}

POSITIVE_COL_PREFIXES = ("tau_", "dt_")


def stable_label(prefix: str, value: object, width: int = 4) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{int(digest[:10], 16) % (10**width):0{width}d}"


def scramble_numeric(series: pd.Series, col: str) -> pd.Series:
    out = series.copy()
    mask = out.notna()
    if not mask.any():
        return out

    vals = out.loc[mask].to_numpy()
    shuffled = rng.permutation(vals)

    if col in COUNT_COLS:
        out.loc[mask] = np.rint(shuffled).astype(int)
        return out

    spread = np.nanstd(vals)
    if not np.isfinite(spread) or spread == 0:
        spread = max(abs(float(np.nanmean(vals))) * 0.05, 1e-6)

    noisy = shuffled + rng.normal(0, spread * 0.05, size=len(shuffled))

    if col in FRACTION_COLS:
        noisy = np.clip(noisy, 0.0, 1.0)
    elif col.startswith(POSITIVE_COL_PREFIXES):
        noisy = np.clip(noisy, 0.0, None)

    out.loc[mask] = noisy
    return out


def postprocess_summary_systems(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for mean_col, lo_col, hi_col, upper in [
        ("binding_frac_mean", "binding_frac_ci_lo", "binding_frac_ci_hi", 1.0),
        ("tau_ps_mean", "tau_ps_ci_lo", "tau_ps_ci_hi", None),
    ]:
        if {mean_col, lo_col, hi_col}.issubset(df.columns):
            mean = df[mean_col].clip(lower=0, upper=upper)
            lo_width = (df[lo_col] - mean).abs()
            hi_width = (df[hi_col] - mean).abs()
            df[mean_col] = mean
            df[lo_col] = (mean - lo_width).clip(lower=0, upper=upper)
            df[hi_col] = (mean + hi_width).clip(lower=0, upper=upper)

    return df


def scramble_df(df: pd.DataFrame, name_map: dict[str, str], kind: str) -> pd.DataFrame:
    df = df.copy()

    if SYSTEM_COL in df.columns:
        df[SYSTEM_COL] = df[SYSTEM_COL].map(name_map).fillna(df[SYSTEM_COL])

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = scramble_numeric(df[col], col)

    for col in df.select_dtypes(include=[object, "string"]).columns:
        if col == SYSTEM_COL:
            continue
        if col == "atom":
            # Keep element-like prefixes so the dashboard can derive atom elements.
            df[col] = [f"C{i + 1}" for i in range(len(df))]
        else:
            df[col] = df[col].map(lambda value: stable_label("COMP", value) if pd.notna(value) else value)

    if kind == "systems":
        df = postprocess_summary_systems(df)

    return df


def main() -> None:
    sys_path = REAL_DATA / "summaries/summary_systems.csv"
    ana_path = REAL_DATA / "summaries/summary_analytes.csv"
    atm_path = REAL_DATA / "summaries/summary_atoms.parquet"

    sys_df = pd.read_csv(sys_path)
    ana_df = pd.read_csv(ana_path)
    atm_df = pd.read_parquet(atm_path)

    print("summary_systems  columns:", sys_df.columns.tolist())
    print("summary_analytes columns:", ana_df.columns.tolist())
    print("summary_atoms    columns:", atm_df.columns.tolist())

    if SYSTEM_COL not in sys_df.columns:
        raise KeyError(f"Expected `{SYSTEM_COL}` in summary_systems.csv")

    real_systems = sys_df[SYSTEM_COL].dropna().unique()
    fake_systems = [f"SYS_{i:03d}" for i in range(len(real_systems))]
    name_map = dict(zip(real_systems, fake_systems))

    print(f"\nMapped {len(name_map)} system names: {real_systems[0]} -> {fake_systems[0]}")

    out_sum = OUT_DATA / "summaries"
    out_sum.mkdir(parents=True, exist_ok=True)

    scramble_df(sys_df, name_map, "systems").to_csv(out_sum / "summary_systems.csv", index=False)
    scramble_df(ana_df, name_map, "analytes").to_csv(out_sum / "summary_analytes.csv", index=False)
    scramble_df(atm_df, name_map, "atoms").to_parquet(out_sum / "summary_atoms.parquet", index=False)

    print("\nSummary files written.")

    mol2_copied = 0
    missing_mol2 = 0
    for real_name, fake_name in name_map.items():
        src_mol2 = REAL_DATA / real_name / "LIG.mol2"
        dst_dir = OUT_DATA / fake_name
        dst_mol2 = dst_dir / "LIG.mol2"
        if src_mol2.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mol2, dst_mol2)
            dst_mol2.chmod(0o644)
            mol2_copied += 1
        else:
            missing_mol2 += 1

    print(f"Copied {mol2_copied} LIG.mol2 files under fake system names.")
    print(f"Missing {missing_mol2} LIG.mol2 files.")
    print("\nDone. Review data/_50_/ before committing.")


if __name__ == "__main__":
    main()
