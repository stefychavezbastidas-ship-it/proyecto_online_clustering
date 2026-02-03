import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
FEATURES_DIR = "features_out"

DATASETS = {
    "animals": 3,
    "fruits": 4
}

DESCRIPTORS = [
    "Momentos de Hu",
    "HOG",
    "SIFT",
    "Embeddings CNN"
]

st.set_page_config(
    page_title="Online Constrained Clustering",
    layout="wide"
)

st.title("Online Constrained K-Means con Descriptores Visuales")
st.markdown(
    "Aplicación para evaluar **clustering no supervisado** "
    "usando **descriptores visuales previamente extraídos**."
)

# =========================
# FUNCIONES DE CARGA
# =========================
def load_csv_features(descriptor, dataset):
    file_map = {
        "Momentos de Hu": f"X_hu_{dataset}.csv",
        "HOG": f"X_hog_{dataset}.csv",
        "SIFT": f"X_sift_{dataset}.csv",
    }

    path = os.path.join(FEATURES_DIR, file_map[descriptor])

    if not os.path.exists(path):
        st.error(f"No existe el archivo {path}")
        st.stop()

    df = pd.read_csv(path)

    meta_cols = ["dataset", "class", "file"]
    X = df.drop(columns=meta_cols, errors="ignore").values
    y = df["class"].values if "class" in df.columns else None

    return X, y, df


def load_embeddings_npz(dataset):
    path = os.path.join(FEATURES_DIR, f"X_emb_{dataset}.npz")

    if not os.path.exists(path):
        st.error(f"No existe el archivo {path}")
        st.stop()

    data = np.load(path, allow_pickle=True)

    if "X" in data.files:
        X = data["X"]
    elif "embeddings" in data.files:
        X = data["embeddings"]
    else:
        st.error("El NPZ no contiene embeddings válidos")
        st.stop()

    return X, None, None


def load_features(descriptor, dataset):
    if descriptor == "Embeddings CNN":
        return load_embeddings_npz(dataset)
    else:
        return load_csv_features(descriptor, dataset)


def plot_pca(X, labels):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": labels.astype(str)
    })

    st.scatter_chart(
        df_pca,
        x="PC1",
        y="PC2",
        color="Cluster"
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.header(" Configuración")

dataset = st.sidebar.selectbox(
    "Dataset",
    list(DATASETS.keys())
)

descriptor = st.sidebar.selectbox(
    "Descriptor",
    DESCRIPTORS
)

k = DATASETS[dataset]

st.sidebar.markdown(f"**Clusters fijos:** {k}")

run = st.sidebar.button(" Ejecutar Clustering")

# =========================
# MAIN
# =========================
if run:
    st.subheader("Carga de Features")

    X, y_true, df = load_features(descriptor, dataset)

    st.success(f"Datos cargados correctamente")
    st.write(f"🔹 Muestras: {X.shape[0]}")
    st.write(f"🔹 Dimensión: {X.shape[1]}")

    # =========================
    # ESCALADO
    # =========================
    X = StandardScaler().fit_transform(X)

    # =========================
    # CLUSTERING
    # =========================
    
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # =========================
    # MÉTRICAS
    # =========================
    st.subheader(" Métricas")

    sil = silhouette_score(X, labels)
    st.metric("Silhouette Score", f"{sil:.4f}")

    # =========================
    # DISTRIBUCIÓN
    # =========================
    st.subheader("Tamaño de Clusters")

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        st.write(f"Cluster {u}: {c} muestras")

    # =========================
    # PCA
    # =========================
    st.subheader(" Visualización PCA (2D)")
    plot_pca(X, labels)

 
