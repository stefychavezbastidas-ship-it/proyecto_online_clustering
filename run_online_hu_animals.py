import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from online_constrained_kmeans import OnlineConstrainedKMeans

CSV_PATH = os.path.join("features_out", "X_hu_animals.csv")

def main():
    df = pd.read_csv(CSV_PATH)

    # Features HU
    hu_cols = [c for c in df.columns if c.startswith("hu_")]
    X = df[hu_cols].values.astype(np.float32)

    # Ground truth
    y = df["class"].astype("category").cat.codes.values

    # Escalado
    Xs = StandardScaler().fit_transform(X)

    classes = df["class"].astype("category")
    k = classes.nunique()

    # Restricción por tamaño (ground truth)
    max_sizes = classes.value_counts().sort_index().values.tolist()

    model = OnlineConstrainedKMeans(
        n_clusters=k,
        max_sizes=max_sizes,
        random_state=42
    )

    pred = model.partial_fit_predict(Xs)

    print("=== ONLINE CONSTRAINED KMEANS | HU ANIMALS ===")
    print("k:", k)
    print("max_sizes:", max_sizes)
    print("sizes_observed:", np.bincount(pred, minlength=k).tolist())

    print("\nInternas:")
    print("Silhouette:", silhouette_score(Xs, pred))
    print("Davies-Bouldin:", davies_bouldin_score(Xs, pred))

    print("\nExternas:")
    print("ARI:", adjusted_rand_score(y, pred))
    print("NMI:", normalized_mutual_info_score(y, pred))

if __name__ == "__main__":
    main()
