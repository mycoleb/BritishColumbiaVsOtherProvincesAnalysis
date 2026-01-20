from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/province_year_features.csv")
OUT_DIR = Path("outputs")


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
    
    # Filter for unemployment rate
    d = df[df["Labour force characteristics"] == "Unemployment rate"].copy()
    print(f"    After filtering for 'Unemployment rate': {len(d):,} rows")
    
    # Filter for 'Estimate' in Statistics
    d = d[d["Statistics"] == "Estimate"].copy()
    print(f"    After filtering for 'Estimate': {len(d):,} rows")
    
    # Convert dates to year
    ref = d["REF_DATE"].astype(str)
    d["year"] = ref.str.slice(0, 4).astype(int)
    
    # Keep just provinces/territories
    d = d[d["GEO"].isin(PROVINCE_NAME_MAP.keys())].copy()
    print(f"    After filtering for provinces: {len(d):,} rows")
    
    # Filter for total gender (not split by men/women)
    if "Gender" in d.columns:
        before = len(d)
        d = d[d["Gender"] == "Total - Gender"].copy()
        print(f"    After filtering for 'Total - Gender': {len(d):,} rows (was {before:,})")
    
    # Filter for all ages (15 years and over)
    if "Age group" in d.columns:
        before = len(d)
        d = d[d["Age group"] == "15 years and over"].copy()
        print(f"    After filtering for '15 years and over': {len(d):,} rows (was {before:,})")
    
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
    
    # Filter for median income
    if "Statistics" in d.columns:
        before = len(d)
        d = d[d["Statistics"] == "Median income"]
        print(f"    After filtering Statistics for 'Median income': {len(d):,} rows (was {before:,})")
    
    # Filter for after-tax income concept
    if "Income concept" in d.columns and len(d) > 0:
        before = len(d)
        d = d[d["Income concept"] == "After-tax income"]
        print(f"    After filtering Income concept for 'After-tax income': {len(d):,} rows (was {before:,})")
    
    # Filter for both sexes
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


def build_features() -> pd.DataFrame:
    """Build the complete feature dataset."""
    print("=" * 80)
    print("STEP 1: Building feature tables...")
    print("=" * 80)
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
        return pd.DataFrame()
    
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
        return pd.DataFrame()
    
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
    
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved features to {PROCESSED_PATH} with {len(df):,} rows")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Provinces: {sorted(df['province'].unique())}")
    
    return df


def cluster_and_plot(
    df: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2024,
    k: int = 4,
    aggregate_to_province: bool = True,
) -> None:
    """Run clustering analysis and create visualizations."""
    print("\n" + "=" * 80)
    print("STEP 2: Clustering and visualization...")
    print("=" * 80)
    print()
    
    # Filter to year window
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    print(f"Filtered to years {start_year}-{end_year}: {len(df):,} rows")
    
    if aggregate_to_province:
        # One row per province (average over years)
        df_model = df.groupby("province", as_index=False).mean(numeric_only=True)
        print(f"Aggregated to province level: {len(df_model)} provinces")
    else:
        # Province-year points
        df_model = df.copy()
        print(f"Using province-year data: {len(df_model)} observations")
    
    feature_cols = [
        "unemployment_rate",
        "median_after_tax_income",
        "gdp_per_capita",
        "population",
    ]
    
    X = df_model[feature_cols].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # KMeans clusters
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    df_model["cluster"] = km.fit_predict(Xs)
    
    # PCA to 2D for plotting
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    df_model["pc1"] = X2[:, 0]
    df_model["pc2"] = X2[:, 1]
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- PCA scatter plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot all points
    scatter = ax.scatter(df_model["pc1"], df_model["pc2"], c=df_model["cluster"], 
                        cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    
    # Label each point with province (and year if not aggregated)
    for _, r in df_model.iterrows():
        label = r["province"] if aggregate_to_province else f'{r["province"]} {int(r["year"])}'
        ax.text(r["pc1"], r["pc2"], label, fontsize=9, ha='center', va='bottom')
    
    # Highlight BC with a special marker
    bc = df_model[df_model["province"] == "British Columbia"]
    if len(bc) > 0:
        ax.scatter(bc["pc1"], bc["pc2"], s=300, marker="*", 
                  facecolors="red", edgecolors="darkred", linewidths=2,
                  label="British Columbia", zorder=5)
        ax.set_title("PCA (2D) of Province Similarity  BC highlighted", fontsize=14, fontweight='bold')
    else:
        ax.set_title("PCA (2D) of Province Similarity", fontsize=14, fontweight='bold')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    pca_path = OUT_DIR / "pca_clusters-minusgender.png"
    fig.savefig(pca_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" Wrote {pca_path}")
    
    # --- Dendrogram (hierarchical clustering) ---
    Z = linkage(Xs, method="ward")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    labels = df_model["province"].tolist() if aggregate_to_province else [
        f'{p} {int(y)}' for p, y in zip(df_model["province"], df_model["year"])
    ]
    
    dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Distance", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    dendro_path = OUT_DIR / "dendrogramminusgender.png"
    fig.savefig(dendro_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" Wrote {dendro_path}")
    
    # --- "Who is BC closest to?" analysis ---
    if aggregate_to_province and "British Columbia" in df_model["province"].values:
        bc_vec = Xs[df_model["province"].values.tolist().index("British Columbia")]
        dists = np.linalg.norm(Xs - bc_vec, axis=1)
        df_model["dist_to_BC"] = dists
        nearest = df_model[df_model["province"] != "British Columbia"].sort_values("dist_to_BC").head(5)
        
        print("\n" + "=" * 80)
        print("Closest provinces to BC (lower distance = more similar):")
        print("=" * 80)
        for idx, row in nearest.iterrows():
            print(f"  {row['province']:25s} - Distance: {row['dist_to_BC']:.3f} (Cluster {int(row['cluster'])})")
        
        # Print feature comparison
        print("\n" + "=" * 80)
        print("Feature comparison (BC vs. most similar provinces):")
        print("=" * 80)
        bc_features = df_model[df_model["province"] == "British Columbia"][feature_cols].iloc[0]
        print(f"\nBritish Columbia:")
        for feat in feature_cols:
            print(f"  {feat:30s}: {bc_features[feat]:,.2f}")
        
        for idx, row in nearest.head(3).iterrows():
            print(f"\n{row['province']}:")
            for feat in feature_cols:
                diff = ((row[feat] - bc_features[feat]) / bc_features[feat]) * 100
                print(f"  {feat:30s}: {row[feat]:,.2f} ({diff:+.1f}% vs BC)")


def main() -> None:
    print("\n")
    print("" + "=" * 78 + "")
    print("" + " " * 20 + "BC PROVINCE CLUSTERING ANALYSIS" + " " * 26 + "")
    print("" + "=" * 78 + "")
    print()
    
    # Step 1: Build features
    df = build_features()
    
    if len(df) == 0:
        print("\nERROR: Failed to build feature dataset. Exiting.")
        return
    
    # Step 2: Cluster and plot
    cluster_and_plot(
        df,
        start_year=2012,  # Use full available range
        end_year=2023,
        k=4,
        aggregate_to_province=True
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {OUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
