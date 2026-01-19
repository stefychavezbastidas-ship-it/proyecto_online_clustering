import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from online_constrained_kmeans import OnlineConstrainedKMeans

CSV_PATH = os.path.join("features_out", "X_hog_fruits.csv")

def main():
    df = pd.read_csv(CSV_PATH)

    hog_cols = [c for c in df.columns if c.startswith("hog_")]
    X = df[hog_cols].values.astype(np.float32)

    y = df["class"].astype("category").cat.codes.values
    classes = df["class"].astype("category")
    k = classes.nunique()

    # Escalado
    Xs = StandardScaler().fit_transform(X)

    # PCA para bajar dimensión (recomendado para HOG)
    pca = PCA(n_components=100, random_state=42)
    Xp = pca.fit_transform(Xs)

    max_sizes = classes.value_counts().sort_index().values.tolist()

    model = OnlineConstrainedKMeans(n_clusters=k, max_sizes=max_sizes, random_state=42)
    pred = model.partial_fit_predict(Xp)

    print("=== ONLINE CONSTRAINED KMEANS | HOG FRUITS (PCA=100) ===")
    print("k:", k)
    print("max_sizes:", max_sizes)
    print("sizes_observed:", np.bincount(pred, minlength=k).tolist())
    print("X shape:", X.shape, "| Xp shape:", Xp.shape)

    print("\nInternas:")
    print("Silhouette:", silhouette_score(Xp, pred))
    print("Davies-Bouldin:", davies_bouldin_score(Xp, pred))

    print("\nExternas:")
    print("ARI:", adjusted_rand_score(y, pred))
    print("NMI:", normalized_mutual_info_score(y, pred))

if __name__ == "__main__":
    main()
