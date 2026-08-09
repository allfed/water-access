"""Aggregate per-country handpump-share estimates into global lose_fraction ranges.

Reads data/original_data/borehole_pump_type_estimates.csv (handpump share among
borehole users, % 0-100, low/high bounds per stratum) and weights each country
by its borehole-dependent population from data/processed/merged_data.csv
(Population x urban share x URBANBorehole, and the rural counterpart).

The HEMP-scenario lose fraction for a stratum is 1 - handpump_share:
handpump users keep their source when electric pumping fails, everyone else
on a motorized borehole is displaced. See
ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md for provenance and scenario variants.

Usage:
    python scripts/Data_Manipulation_Scripts/aggregate_borehole_pump_estimates.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ESTIMATES_PATH = REPO_ROOT / "data/original_data/borehole_pump_type_estimates.csv"
MERGED_DATA_PATH = REPO_ROOT / "data/processed/merged_data.csv"


def load_weighted_estimates() -> pd.DataFrame:
    est = pd.read_csv(ESTIMATES_PATH)
    md = pd.read_csv(MERGED_DATA_PATH)
    md = md.assign(
        urban_bh_pop=md["Population"]
        * md["% urban"]
        / 100
        * md["URBANBorehole"].fillna(0)
        / 100,
        rural_bh_pop=md["Population"]
        * (1 - md["% urban"] / 100)
        * md["RURALBorehole"].fillna(0)
        / 100,
    )
    return est.drop(columns=["urban_bh_pop", "rural_bh_pop"]).merge(
        md[["alpha3", "urban_bh_pop", "rural_bh_pop"]], on="alpha3", how="left"
    )


def weighted_bounds(df: pd.DataFrame, stratum: str) -> tuple[float, float]:
    """Population-weighted (low, high) handpump share for a stratum, in %."""
    w = df[f"{stratum.lower()}_bh_pop"]
    lo = (df[f"{stratum.upper()}_handpump_low"] * w).sum() / w.sum()
    hi = (df[f"{stratum.upper()}_handpump_high"] * w).sum() / w.sum()
    return lo, hi


def main() -> None:
    df = load_weighted_estimates()
    for stratum in ("urban", "rural"):
        lo, hi = weighted_bounds(df, stratum)
        cov = (
            df.loc[df["origin"] == "researched", f"{stratum}_bh_pop"].sum()
            / df[f"{stratum}_bh_pop"].sum()
        )
        print(
            f"{stratum.upper()}: handpump share {lo:.1f}-{hi:.1f}% of borehole users"
            f" -> HEMP {stratum}_borehole_lose_fraction {1 - hi / 100:.3f}-{1 - lo / 100:.3f}"
            f" (researched-country coverage {cov:.1%})"
        )


if __name__ == "__main__":
    main()
