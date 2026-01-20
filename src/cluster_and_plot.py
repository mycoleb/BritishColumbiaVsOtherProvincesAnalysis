from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


DATA_PATH = Path("data/processed/province_year_features.csv")
OUT_DIR = Path("outputs")


def main(
    start_year: int = 2010,
    end_year: int = 2024,
    k: int = 4,
    aggregate_to_province: bool = True,
) -> None:
    df = pd.read_csv(DATA_PATH)

    # Filter to year window
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    if aggregate_to_province:
        # One row per province (average over years) -> simpler â€œwhich province is like BCâ€
        df_model = df.groupby("province", as_index=False).mean(numeric_only=True)
        id_cols = ["province"]
    else:
        # Province-year points (shows how similarity shifts over time)
        df_model = df.copy()
        id_cols = ["province", "year"]

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
    fig = plt.figure()
    ax = plt.gca()

    # Plot all points
    ax.scatter(df_model["pc1"], df_model["pc2"])

    # Label each point with province (and year if not aggregated)
    for _, r in df_model.iterrows():
        label = r["province"] if aggregate_to_province else f'{r["province"]} {int(r["year"])}'
        ax.text(r["pc1"], r["pc2"], label, fontsize=8)

    # Highlight BC
    bc = df_model[df_model["province"] == "British Columbia"]
    if len(bc) > 0:
        ax.scatter(bc["pc1"], bc["pc2"], s=180, marker="o", facecolors="none")
        ax.set_title("PCA (2D) of Province Similarity â€” BC highlighted")
    else:
        ax.set_title("PCA (2D) of Province Similarity")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    pca_path = OUT_DIR / "pca_clusters.png"
    fig.savefig(pca_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pca_path}")

    # --- Dendrogram (hierarchical clustering) ---
    Z = linkage(Xs, method="ward")
    fig = plt.figure()
    ax = plt.gca()

    labels = df_model["province"].tolist() if aggregate_to_province else [
        f'{p} {int(y)}' for p, y in zip(df_model["province"], df_model["year"])
    ]

    dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)")
    ax.set_ylabel("Distance")

    dendro_path = OUT_DIR / "dendrogram.png"
    fig.savefig(dendro_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dendro_path}")

    # --- â€œWho is BC closest to?â€ quick printout ---
    if aggregate_to_province and "British Columbia" in df_model["province"].values:
        # Distance to BC in standardized feature space
        bc_vec = Xs[df_model["province"].values.tolist().index("British Columbia")]
        dists = np.linalg.norm(Xs - bc_vec, axis=1)
        df_model["dist_to_BC"] = dists
        nearest = df_model[df_model["province"] != "British Columbia"].sort_values("dist_to_BC").head(5)
        print("\nClosest provinces to BC (lower = more similar):")
        print(nearest[["province", "dist_to_BC", "cluster"]].to_string(index=False))


if __name__ == "__main__":
    main()