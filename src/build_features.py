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
    print("  Processing unemployment data...")
    df = _normalize_cols(pd.read_csv(RAW_DIR / "unemployment_monthly.csv", low_memory=False))
    
    print(f"    Loaded {len(df):,} rows")
    
    # Filter for unemployment rate in 'Labour force characteristics' column
    d = df[df["Labour force characteristics"] == "Unemployment rate"].copy()
    print(f"    After filtering for 'Unemployment rate': {len(d):,} rows")
    
    if len(d) == 0:
        print("    ERROR: No unemployment rate data found!")
        return pd.DataFrame()
    
    # Filter for 'Estimate' in Statistics (not standard error)
    if "Statistics" in d.columns:
        before = len(d)
        d = d[d["Statistics"] == "Estimate"].copy()
        print(f"    After filtering for 'Estimate': {len(d):,} rows (was {before:,})")
    
    # Convert dates to year
    ref = d["REF_DATE"].astype(str)
    d["year"] = ref.str.slice(0, 4).astype(int)
    
    # Keep just provinces/territories
    d = d[d["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    print(f"    After filtering for provinces: {len(d):,} rows")
    
    # Check what Gender values exist and filter appropriately
    if "Gender" in d.columns:
        gender_values = d["Gender"].unique()
        print(f"    Unique Gender values: {gender_values[:5]}")
        before = len(d)
        # Try different possible values for "all genders"
        d = d[d["Gender"].isin(["Both sexes", "Both genders", "Total, all genders"])].copy()
        if len(d) == 0:
            # If no match, just take all (some tables don't have gender breakdown)
            print(f"    Warning: No gender filter match, keeping all rows")
            d = df[df["Labour force characteristics"] == "Unemployment rate"].copy()
            d = d[d["Statistics"] == "Estimate"].copy()
            d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)
            d = d[d["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
        else:
            print(f"    After filtering Gender: {len(d):,} rows (was {before:,})")
    
    # Check what Age group values exist and filter appropriately
    if "Age group" in d.columns:
        age_values = d["Age group"].unique()
        print(f"    Unique Age group values: {age_values[:5]}")
        before = len(d)
        # Try different possible values for "all ages"
        d = d[d["Age group"].isin(["15 years and over", "15 years and older", "Total, 15 years and over"])].copy()
        if len(d) == 0:
            print(f"    Warning: No age filter match, skipping age filter")
        else:
            print(f"    After filtering Age group: {len(d):,} rows (was {before:,})")
    
    if len(d) == 0:
        print("    ERROR: No data remains after filtering!")
        return pd.DataFrame()
    
    # Annual mean unemployment rate
    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "unemployment_rate"})
    )
    print(f"    Final output: {len(out):,} rows, years {out['year'].min()}-{out['year'].max()}")
    return out


def build_population_annual() -> pd.DataFrame:
    """From 17-10-0005-01 -> total population by province/year."""
    print("  Processing population data...")
    df = _normalize_cols(pd.read_csv(RAW_DIR / "population.csv", low_memory=False))
    
    print(f"    Loaded {len(df):,} rows")
    
    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    print(f"    After filtering for provinces: {len(d):,} rows")
    
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)
    
    # Filter for totals
    for col in ["Age group", "Gender", "Sex"]:
        if col in d.columns:
            before = len(d)
            d = d[d[col].astype(str).str.contains("All ages|Total", case=False, na=False)]
            print(f"    After filtering {col} for totals: {len(d):,} rows (was {before:,})")
    
    if len(d) == 0:
        print("    ERROR: No data remains after filtering!")
        return pd.DataFrame()
    
    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "population"})
    )
    print(f"    Final output: {len(out):,} rows, years {out['year'].min()}-{out['year'].max()}")
    return out


def build_gdp_annual() -> pd.DataFrame:
    """From 36-10-0402-01 -> total GDP (all industries) by province/year."""
    print("  Processing GDP data...")
    df = _normalize_cols(pd.read_csv(RAW_DIR / "gdp_by_industry.csv", low_memory=False))
    
    print(f"    Loaded {len(df):,} rows")
    
    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    print(f"    After filtering for provinces: {len(d):,} rows")
    
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)
    
    # Filter for "All industries"
    naics_col = "North American Industry Classification System (NAICS)"
    if naics_col in d.columns:
        before = len(d)
        d = d[d[naics_col].astype(str).str.contains("All industries", case=False, na=False)]
        print(f"    After filtering for 'All industries': {len(d):,} rows (was {before:,})")
    
    if len(d) == 0:
        print("    ERROR: No data remains after filtering!")
        return pd.DataFrame()
    
    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "gdp_total"})
    )
    print(f"    Final output: {len(out):,} rows, years {out['year'].min()}-{out['year'].max()}")
    return out


def build_income_annual() -> pd.DataFrame:
    """From 11-10-0091-01 -> median after-tax income by province/year."""
    print("  Processing income data...")
    df = _normalize_cols(pd.read_csv(RAW_DIR / "income.csv", low_memory=False))
    
    print(f"    Loaded {len(df):,} rows")
    
    d = df[df["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    print(f"    After filtering for provinces: {len(d):,} rows")
    
    d["year"] = d["REF_DATE"].astype(str).str.slice(0, 4).astype(int)
    
    # Filter for median income (exact match)
    if "Statistics" in d.columns:
        before = len(d)
        d = d[d["Statistics"] == "Median income"]
        print(f"    After filtering Statistics for 'Median income': {len(d):,} rows (was {before:,})")
    
    # Filter for after-tax income concept (exact match)
    if "Income concept" in d.columns and len(d) > 0:
        before = len(d)
        d = d[d["Income concept"] == "After-tax income"]
        print(f"    After filtering Income concept for 'After-tax income': {len(d):,} rows (was {before:,})")
    
    # Filter for both sexes (exact match)
    if "Sex" in d.columns and len(d) > 0:
        before = len(d)
        d = d[d["Sex"] == "Both sexes"]
        print(f"    After filtering Sex for 'Both sexes': {len(d):,} rows (was {before:,})")
    
    # Filter for all persons demographics
    if "Demographic characteristics" in d.columns and len(d) > 0:
        before = len(d)
        d = d[d["Demographic characteristics"].astype(str).str.contains("All persons|Persons in economic families", case=False, na=False)]
        print(f"    After filtering Demographic characteristics: {len(d):,} rows (was {before:,})")
    
    if len(d) == 0:
        print("    ERROR: No data remains after filtering!")
        return pd.DataFrame()
    
    out = (
        d.groupby(["GEO", "year"], as_index=False)["VALUE"]
        .mean()
        .rename(columns={"GEO": "province", "VALUE": "median_after_tax_income"})
    )
    print(f"    Final output: {len(out):,} rows, years {out['year'].min()}-{out['year'].max()}")
    return out


def main() -> None:
    print("Building feature tables...")
    print()
    
    u = build_unemployment_annual()
    print()
    p = build_population_annual()
    print()
    g = build_gdp_annual()
    print()
    i = build_income_annual()
    print()
    
    if len(u) == 0 or len(p) == 0 or len(g) == 0 or len(i) == 0:
        print("ERROR: One or more feature tables are empty!")
        print(f"  Unemployment: {len(u)} rows")
        print(f"  Population: {len(p)} rows")
        print(f"  GDP: {len(g)} rows")
        print(f"  Income: {len(i)} rows")
        return
    
    # Merge on province + year
    print("Merging tables...")
    df = u.merge(p, on=["province", "year"], how="inner")
    print(f"  After merging unemployment + population: {len(df):,} rows")
    
    df = df.merge(g, on=["province", "year"], how="inner")
    print(f"  After merging + GDP: {len(df):,} rows")
    
    df = df.merge(i, on=["province", "year"], how="inner")
    print(f"  After merging + income: {len(df):,} rows")
    
    if len(df) == 0:
        print("ERROR: No rows remain after merging!")
        print("This likely means the year ranges don't overlap across all datasets.")
        print(f"  Unemployment years: {u['year'].min()}-{u['year'].max()}")
        print(f"  Population years: {p['year'].min()}-{p['year'].max()}")
        print(f"  GDP years: {g['year'].min()}-{g['year'].max()}")
        print(f"  Income years: {i['year'].min()}-{i['year'].max()}")
        return
    
    # Derived features
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
    print(f"\n{'='*80}")
    print(f"SUCCESS! Wrote {OUT_PATH} with {len(df):,} rows")
    print(f"{'='*80}")
    
    if len(df) > 0:
        print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
        print(f"Provinces: {len(df['province'].unique())} provinces")
        print(f"  {sorted(df['province'].unique())}")
        print(f"\nSample data (first 5 rows):")
        print(df.head().to_string())


if __name__ == "__main__":
    main()