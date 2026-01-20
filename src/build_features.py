from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/province_year_features.csv")


PROVINCE_NAME_MAP = {
    "British Columbia": "British Columbia",
    "Ontario": "Ontario",
    "Quebec": "Quebec",
    "Alberta": "Alberta",
    "Manitoba": "Manitoba",
    "Saskatchewan": "Saskatchewan",
    "New Brunswick": "New Brunswick",
    "Nova Scotia": "Nova Scotia",
    "Prince Edward Island": "Prince Edward Island",
    "Newfoundland and Labrador": "Newfoundland and Labrador",
    # You can keep territories too if you want:
    "Yukon": "Yukon",
    "Northwest Territories": "Northwest Territories",
    "Nunavut": "Nunavut",
}


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def build_unemployment_annual() -> pd.DataFrame:
    """From 14-10-0287-03 monthly unemployment rate -> annual average by province."""
    df = _normalize_cols(pd.read_csv(RAW_DIR / "unemployment_monthly.csv", low_memory=False))

    # Typical StatCan columns you’ll see: "REF_DATE", "GEO", "Sex", "Age group", "Statistics", "VALUE", etc.
    needed = {"REF_DATE", "GEO", "Statistics", "VALUE"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Unemployment table missing columns: {missing}")

    # Filter to unemployment rate (the table contains multiple stats like employment level, participation, etc.)
    d = df[df["Statistics"].astype(str).str.contains("Unemployment rate", case=False, na=False)].copy()

    # Convert dates to year (monthly REF_DATE often like "2024-12" or "2024-12-01" depending on table)
    ref = d["REF_DATE"].astype(str)
    d["year"] = ref.str.slice(0, 4).astype(int)

    # Keep just provinces/territories we recognize
    d = d[d["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()

    # Annual mean unemployment rate
    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "unemployment_rate"})
    )
    return out


def build_population_annual() -> pd.DataFrame:
    """From 17-10-0005-01 -> total population by province/year."""
    df = _normalize_cols(pd.read_csv(RAW_DIR / "population.csv", low_memory=False))

    needed = {"REF_DATE", "GEO", "VALUE"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Population table missing columns: {missing}")

    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)

    # Try to isolate "Total population" rows if the table includes age groups / genders.
    # This is a heuristic: you may need to adjust these filters after a quick peek at the CSV columns.
    for col in ["Age group", "Gender", "Sex"]:
        if col in d.columns:
            d = d[d[col].astype(str).str.contains("All|Both|Total", case=False, na=False)]

    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "population"})
    )
    return out


def build_gdp_annual() -> pd.DataFrame:
    """From 36-10-0402-01 -> total GDP (all industries) by province/year."""
    df = _normalize_cols(pd.read_csv(RAW_DIR / "gdp_by_industry.csv", low_memory=False))

    needed = {"REF_DATE", "GEO", "VALUE"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"GDP table missing columns: {missing}")

    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)

    # Heuristic: keep “All industries” / “Total” if there is an industry dimension.
    for col in ["Industry", "North American Industry Classification System (NAICS)"]:
        if col in d.columns:
            d = d[d[col].astype(str).str.contains("All|Total", case=False, na=False)]

    # Heuristic: prefer current dollars if present
    for col in ["Prices", "UOM", "Unit of measure"]:
        if col in d.columns:
            # don’t over-filter unless you see multiple representations;
            # safe default: keep all, then group-average.
            pass

    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "gdp_total"})
    )
    return out


def build_income_annual() -> pd.DataFrame:
    """From 11-10-0091-01 -> median after-tax income by province/year (heuristic filters)."""
    df = _normalize_cols(pd.read_csv(RAW_DIR / "income.csv", low_memory=False))

    needed = {"REF_DATE", "GEO", "VALUE"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Income table missing columns: {missing}")

    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)

    # Heuristics: keep median after-tax income, all persons
    for col in ["Income concept", "Income statistics", "Characteristics", "Selected demographic characteristics"]:
        if col in d.columns:
            if d[col].astype(str).str.contains("after-tax", case=False, na=False).any():
                d = d[d[col].astype(str).str.contains("after-tax", case=False, na=False)]
            if d[col].astype(str).str.contains("median", case=False, na=False).any():
                d = d[d[col].astype(str).str.contains("median", case=False, na=False)]

    for col in ["Gender", "Sex", "Age group"]:
        if col in d.columns:
            d = d[d[col].astype(str).str.contains("All|Both|Total|15 years and over", case=False, na=False)]

    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "median_after_tax_income"})
    )
    return out


def main() -> None:
    print("Building feature tables...")
    u = build_unemployment_annual()
    p = build_population_annual()
    g = build_gdp_annual()
    i = build_income_annual()

    # Merge on province + year
    df = u.merge(p, on=["province", "year"], how="inner")
    df = df.merge(g, on=["province", "year"], how="inner")
    df = df.merge(i, on=["province", "year"], how="inner")

    # Derived features that help similarity
    df["gdp_per_capita"] = df["gdp_total"] / df["population"].replace(0, np.nan)

    # Keep a clean set
    keep = [
        "province",
        "year",
        "unemployment_rate",
        "population",
        "gdp_total",
        "median_after_tax_income",
        "gdp_per_capita",
    ]
    df = df[keep].dropna()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df):,} rows")


if __name__ == "__main__":
    main()
