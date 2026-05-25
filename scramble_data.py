#!/usr/bin/env python3
"""
Generate fake-but-realistic showcase data for the Streamlit dashboard.

Run once from the repository root. The script reads the real local `_50_`
summary data, anonymizes system identifiers, perturbs metrics, and writes
synthetic LIG.mol2 files into matching fake system folders.
"""
from __future__ import annotations

import hashlib
import os
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


def write_synthetic_mol2(path: Path, seed_index: int) -> None:
    """Write a small valid synthetic molecule instead of copying real structures."""
    local_rng = np.random.default_rng(SEED + seed_index)
    n_atoms = int(local_rng.integers(8, 18))
    elements = ["C"] * n_atoms
    if n_atoms >= 10:
        elements[2] = "O"
        elements[5] = "N"
    if n_atoms >= 14:
        elements[10] = "S"

    coords = []
    for i in range(n_atoms):
        angle = i * 0.72
        radius = 1.2 + 0.08 * i
        coords.append(
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.18 * np.sin(i),
            )
        )

    bonds = [(i, i + 1) for i in range(1, n_atoms)]
    if n_atoms >= 8:
        bonds.append((1, 6))

    atom_lines = []
    for idx, (element, (x, y, z)) in enumerate(zip(elements, coords), start=1):
        atom_type = {"C": "c3", "O": "oh", "N": "n3", "S": "s3"}.get(element, "c3")
        atom_lines.append(
            f"{idx:7d} {element}{idx:<5d} {x:10.4f} {y:10.4f} {z:10.4f} "
            f"{atom_type:<8s} 1 LIG {0.0:10.6f}"
        )

    bond_lines = [
        f"{idx:6d} {a:5d} {b:5d} 1"
        for idx, (a, b) in enumerate(bonds, start=1)
    ]

    text = "\n".join(
        [
            "@<TRIPOS>MOLECULE",
            "LIG",
            f"{n_atoms:5d} {len(bonds):5d}     1     0     0",
            "SMALL",
            "USER_CHARGES",
            "",
            "@<TRIPOS>ATOM",
            *atom_lines,
            "@<TRIPOS>BOND",
            *bond_lines,
            "@<TRIPOS>SUBSTRUCTURE",
            "     1 LIG         1 TEMP              0 ****  ****    0 ROOT",
            "",
        ]
    )
    path.write_text(text)


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

    mol2_written = 0
    for idx, fake_name in enumerate(name_map.values()):
        dst_dir = OUT_DATA / fake_name
        dst_mol2 = dst_dir / "LIG.mol2"
        dst_dir.mkdir(parents=True, exist_ok=True)
        write_synthetic_mol2(dst_mol2, idx)
        mol2_written += 1

    print(f"Wrote {mol2_written} synthetic LIG.mol2 files under fake system names.")
    print("\nDone. Review data/_50_/ before committing.")


if __name__ == "__main__":
    main()
