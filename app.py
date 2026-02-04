# app.py
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from online_constrained_kmeans import OnlineConstrainedKMeans

FEATURES_DIR = "features_out"

DATASETS = {"animals": 3, "fruits": 4}
DESCRIPTORS = ["Momentos de Hu", "HOG", "SIFT", "Embeddings CNN"]


# ========================= UI helpers =========================
def pretty_label(name: str) -> str:
    table = {
        "cane": "Perro",
        "gatto": "Gato",
        "elefante": "Elefante",
        "Cherry 4": "Cereza",
        "Orange 1": "Naranja",
        "Pineapple 1": "Piña",
        "Strawberry 1": "Fresa",
    }
    return table.get(str(name), str(name))


def pretty_dataset_name(ds: str) -> str:
    return "Animales (3 clases)" if ds == "animals" else "Frutas (4 clases)"


def safe_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


# ========================= Data loading =========================
@dataclass
class LoadedData:
    X: np.ndarray
    y_true: Optional[np.ndarray]
    df_raw: pd.DataFrame
    feature_file: str


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        return ("git-lfs" in head) and ("oid sha256" in head)
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_features(descriptor: str, dataset: str) -> LoadedData:
    file_map = {
        "Momentos de Hu": f"X_hu_{dataset}.csv",
        "HOG": f"X_hog_{dataset}.csv",
        "SIFT": f"X_sift_{dataset}.csv",
        "Embeddings CNN": f"X_emb_{dataset}.csv",
    }
    fname = file_map[descriptor]
    path = os.path.join(FEATURES_DIR, fname)

    if not os.path.exists(path):
        st.error(f"❌ No existe: {path}")
        st.stop()

    if is_lfs_pointer(path):
        st.error(
            f"⚠️ **{fname}** parece un puntero Git LFS (no son datos reales).\n\n"
            "Solución: en el repo clonado ejecuta:\n"
            "1) `git lfs install`\n"
            "2) `git lfs pull`\n"
            "y verifica que el CSV tenga miles de filas."
        )
        st.stop()

    df = read_csv_cached(path)
    meta_cols = ["dataset", "class:"]  # por si algún csv raro
    # columnas estándar del proyecto
    meta_cols2 = ["dataset", "class", "file"]

    # detecta columnas meta presentes
    drop_cols = [c for c in meta_cols2 if c in df.columns]
    if not drop_cols:
        drop_cols = [c for c in meta_cols if c in df.columns]

    X = df.drop(columns=drop_cols, errors="ignore").to_numpy(dtype=np.float32)
    y = df["class"].to_numpy() if "class" in df.columns else None

    return LoadedData(X=X, y_true=y, df_raw=df, feature_file=fname)


def compute_max_sizes(y_true: Optional[np.ndarray], k: int, mode: str, uniform_max: int) -> np.ndarray:
    if mode == "ground_truth":
        if y_true is None:
            raise ValueError("Ground Truth requiere columna 'class'. Usa Uniforme.")
        classes, counts = np.unique(y_true, return_counts=True)
        if len(classes) != k:
            raise ValueError(
                f"Ground Truth tiene {len(classes)} clases, pero k={k}. "
                "Ajusta k o usa Uniforme."
            )
        order = np.argsort(classes.astype(str))
        return counts[order].astype(int)
    return np.full(k, int(uniform_max), dtype=int)


def cluster_to_class_mapping(y_true: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for c in np.unique(labels):
        mask = labels == c
        vals, cnts = np.unique(y_true[mask], return_counts=True)
        mapping[int(c)] = str(vals[int(np.argmax(cnts))])
    return mapping


def update_mapping_online(y_true: Optional[np.ndarray]):
    """Mapping cluster->clase por mayoría con lo ya procesado."""
    if y_true is None:
        st.session_state.mapping = None
        return
    if "labels" not in st.session_state or len(st.session_state.labels) < 2:
        st.session_state.mapping = None
        return
    labels_np = np.array(st.session_state.labels, dtype=int)
    y_np = np.array(y_true[: len(labels_np)], dtype=object)
    st.session_state.mapping = cluster_to_class_mapping(y_np, labels_np)


# ========================= Live feature extraction =========================
@st.cache_resource(show_spinner=False)
def load_resnet18_embedder():
    try:
        import torch
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        preprocess = weights.transforms()
        return ("ok", model, preprocess)
    except Exception as e:
        return ("err", str(e), None)


def embed_cnn_from_pil(img: Image.Image) -> Optional[np.ndarray]:
    status, model_or_err, preprocess = load_resnet18_embedder()
    if status != "ok":
        return None
    import torch

    model = model_or_err
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).squeeze(0).cpu().numpy().astype(np.float32)  # (512,)
    return emb


def _require(module_name: str, install_hint: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        st.error(f"Falta dependencia: **{module_name}**.")
        st.code(install_hint)
        return False


def pil_to_gray(img: Image.Image) -> Optional[np.ndarray]:
    if not _require("cv2", "py -m pip install opencv-python"):
        return None
    import cv2

    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def extract_hu_from_image(img: Image.Image) -> Optional[np.ndarray]:
    if not _require("cv2", "py -m pip install opencv-python"):
        return None
    import cv2

    gray = pil_to_gray(img)
    if gray is None:
        return None
    gray = cv2.resize(gray, (128, 128))
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments).flatten().astype(np.float32)
    # log-scale (estable)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu


def extract_hog_from_image(img: Image.Image) -> Optional[np.ndarray]:
    if not _require("skimage", "py -m pip install scikit-image"):
        return None
    from skimage.feature import hog

    gray = np.array(img.convert("L"))
    gray = np.array(Image.fromarray(gray).resize((128, 128)))
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_sift_from_image(img: Image.Image, target_dim: int) -> Optional[np.ndarray]:
    # SIFT suele requerir opencv-contrib-python
    if not _require("cv2", "py -m pip install opencv-contrib-python"):
        return None
    import cv2

    gray = pil_to_gray(img)
    if gray is None:
        return None
    gray = cv2.resize(gray, (256, 256))

    try:
        sift = cv2.SIFT_create()
    except Exception:
        st.error("Tu OpenCV no trae SIFT. Instala opencv-contrib-python.")
        st.code("py -m pip install opencv-contrib-python")
        return None

    _, des = sift.detectAndCompute(gray, None)

    # vector fijo (mean+std)
    if des is None or len(des) == 0:
        vec = np.zeros(256, dtype=np.float32)
    else:
        des = des.astype(np.float32)
        mu = des.mean(axis=0)
        sd = des.std(axis=0)
        vec = np.concatenate([mu, sd]).astype(np.float32)

    # ajustar a la dimensión del CSV (para poder escalar)
    if vec.shape[0] < target_dim:
        vec = np.pad(vec, (0, target_dim - vec.shape[0]))
    elif vec.shape[0] > target_dim:
        vec = vec[:target_dim]
    return vec.astype(np.float32)


def extract_features_live(descriptor: str, img: Image.Image, target_dim: int) -> Optional[np.ndarray]:
    if descriptor == "Embeddings CNN":
        return embed_cnn_from_pil(img)
    if descriptor == "Momentos de Hu":
        return extract_hu_from_image(img)
    if descriptor == "HOG":
        return extract_hog_from_image(img)
    if descriptor == "SIFT":
        return extract_sift_from_image(img, target_dim=target_dim)
    return None


# ========================= Online core =========================
def init_or_reset_model(k: int, max_sizes: np.ndarray):
    st.session_state.model = OnlineConstrainedKMeans(n_clusters=int(k), max_sizes=max_sizes)
    st.session_state.ptr = 0
    st.session_state.labels = []
    st.session_state.history = []
    st.session_state.mapping = None
    st.session_state.scaler = None


def online_step(X_scaled: np.ndarray, y_true: Optional[np.ndarray]) -> bool:
    if st.session_state.ptr >= len(X_scaled):
        return False

    i = int(st.session_state.ptr)
    x = X_scaled[i]
    y = y_true[i] if y_true is not None else None

    label, info = st.session_state.model.partial_fit_one(x)

    st.session_state.labels.append(int(label))
    st.session_state.history.append(
        {
            "t": int(i),
            "y_true": y,
            "cluster": int(label),
            "tamano_clusters": info["counts"].tolist(),
            "cupo_clusters": info["max_sizes"].tolist(),
            "orden_intento": info["tried_order"],
            "distancias": [float(d) for d in info["dists"]],
            "fallback": bool(info["fallback_used"]),
        }
    )
    st.session_state.ptr += 1
    update_mapping_online(y_true)
    return True


def fit_online_full(X_scaled: np.ndarray, y_true: Optional[np.ndarray]):
    labels = []
    for i in range(len(X_scaled)):
        lab, _ = st.session_state.model.partial_fit_one(X_scaled[i])
        labels.append(int(lab))
    st.session_state.labels = labels
    st.session_state.ptr = len(X_scaled)
    update_mapping_online(y_true)


def can_place_in_cluster(cluster_id: int) -> bool:
    c = st.session_state.model.counts_.astype(int)
    m = st.session_state.model.max_sizes.astype(int)
    return c[cluster_id] < m[cluster_id]


def predict_cluster_no_update(x_scaled: np.ndarray) -> Tuple[int, np.ndarray, bool, List[int]]:
    """
    Predice cluster SIN actualizar centroides ni counts.
    Respeta cupos: elige el cluster más cercano con cupo disponible.
    Si todos están llenos => fallback (más cercano aunque esté lleno).
    """
    model = st.session_state.model
    if model.centroids_ is None:
        # sin centroides aún: inicialización mínima para no romper
        model.centroids_ = np.tile(x_scaled, (model.k, 1)).astype(np.float32)

    dists = np.linalg.norm(model.centroids_ - x_scaled, axis=1)
    order = np.argsort(dists).tolist()
    probs = safe_softmax(-dists)

    chosen = None
    for cid in order:
        if can_place_in_cluster(cid):
            chosen = int(cid)
            break

    fallback = False
    if chosen is None:
        chosen = int(order[0])
        fallback = True

    return chosen, probs.astype(float), fallback, order


def cluster_name(cluster_id: int) -> str:
    mapping = st.session_state.get("mapping", None)
    if mapping and int(cluster_id) in mapping:
        return pretty_label(mapping[int(cluster_id)])
    return f"Cluster {int(cluster_id) + 1}"


def compute_metrics(X_scaled: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    out = {"silhouette": None, "ari": None, "nmi": None}
    if len(np.unique(labels)) > 1:
        out["silhouette"] = float(silhouette_score(X_scaled, labels))
    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    return out


def safe_sample(X: np.ndarray, y: Optional[np.ndarray], labels: np.ndarray, max_n: int = 2000):
    n = min(len(X), len(labels))
    X = X[:n]
    labels = labels[:n]
    y = y[:n] if y is not None else None
    if len(X) <= max_n:
        return X, y, labels
    step = int(np.ceil(len(X) / max_n))
    idx = np.arange(0, len(X), step)
    return X[idx], (y[idx] if y is not None else None), labels[idx]


# ========================= Compare methods (rubrica) =========================
def run_online_for_eval(X_scaled: np.ndarray, k: int, max_sizes: np.ndarray, limit_n: int = 1000) -> np.ndarray:
    n = min(len(X_scaled), int(limit_n))
    model = OnlineConstrainedKMeans(n_clusters=int(k), max_sizes=max_sizes)
    labels = []
    for i in range(n):
        lab, _ = model.partial_fit_one(X_scaled[i])
        labels.append(int(lab))
    return np.array(labels, dtype=int)


def compare_methods(dataset: str, k: int, mode: str, uniform_max: int, limit_n: int = 1000) -> pd.DataFrame:
    rows = []
    for desc in DESCRIPTORS:
        d = load_features(desc, dataset)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(d.X)

        try:
            ms = compute_max_sizes(d.y_true, int(k), mode, int(uniform_max))
        except Exception:
            ms = compute_max_sizes(d.y_true, int(k), "uniform", int(uniform_max))

        labels = run_online_for_eval(Xs, int(k), ms, limit_n=limit_n)
        X_eval = Xs[: len(labels)]
        y_eval = d.y_true[: len(labels)] if d.y_true is not None else None
        mets = compute_metrics(X_eval, labels, y_eval)

        rows.append(
            {
                "Método": desc,
                "N usados": len(labels),
                "Silhouette": None if mets["silhouette"] is None else round(mets["silhouette"], 4),
                "ARI": None if mets["ari"] is None else round(mets["ari"], 4),
                "NMI": None if mets["nmi"] is None else round(mets["nmi"], 4),
            }
        )
    return pd.DataFrame(rows)


# ========================= Page style =========================
st.set_page_config(page_title="Clustering Online", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.1rem;}
h1,h2,h3 {letter-spacing:-0.3px;}
.card {border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.03); border-radius:18px; padding:16px;}
.muted {color:#9aa0a6; font-size:0.92rem;}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.04); font-size:0.85rem;}
.big {font-size:28px; font-weight:900;}
hr {border:none; border-top:1px solid rgba(255,255,255,0.09); margin:14px 0;}
</style>
""",
    unsafe_allow_html=True,
)

# ========================= Sidebar =========================
with st.sidebar:
    st.header("⚙️ Configuración")

    dataset = st.radio("Dataset", ["animals", "fruits"], format_func=pretty_dataset_name)
    descriptor = st.selectbox("Método de extracción", DESCRIPTORS)

    st.subheader("Clustering")
    k_default = int(DATASETS[dataset])
    k = st.number_input("Número de clusters (k)", 2, 15, k_default, 1)

    st.subheader("Restricción de tamaño")
    restr = st.radio("Tipo", ["Uniforme", "Ground Truth"], index=0)
    mode = "uniform" if restr == "Uniforme" else "ground_truth"
    uniform_max = st.number_input("Cupo (Uniforme)", 1, 200000, 50, 1)

    st.markdown("---")
    init_btn = st.button("🚀 Inicializar modelo (dataset completo)", use_container_width=True)
    colA, colB = st.columns(2)
    step_btn = colA.button("+1", use_container_width=True)
    run_all_btn = colB.button("Todo", use_container_width=True)
    reset_btn = st.button("🔁 Reiniciar", use_container_width=True)

# ========================= Header =========================
st.markdown(
    """
<div style="display:flex; align-items:center; gap:12px;">
  <div style="font-size:34px;">🧠</div>
  <div>
    <br>
    <br>
    <div class="big">Clustering Online </div>
    <div class="muted">
      GENERAL
    </div>
  </div>
</div>
<hr/>
""",
    unsafe_allow_html=True,
)

# ========================= Load & setup =========================
data = load_features(descriptor, dataset)
X = data.X
y_true = data.y_true

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    max_sizes = compute_max_sizes(y_true, int(k), mode, int(uniform_max))
except Exception as e:
    st.error(str(e))
    st.stop()

signature = (dataset, descriptor, int(k), mode, int(uniform_max))
if "signature" not in st.session_state:
    st.session_state.signature = None

if reset_btn or st.session_state.signature != signature or "model" not in st.session_state:
    st.session_state.signature = signature
    init_or_reset_model(int(k), max_sizes)
    st.session_state.scaler = scaler

# Inicializar con dataset completo (para mapping y centroides estables)
if init_btn:
    with st.spinner("Procesando TODO el dataset (online)..."):
        st.session_state.scaler = scaler
        fit_online_full(X_scaled, y_true)
    st.success("✅ Listo ya existe cluster.")

# Controles online
if step_btn:
    online_step(X_scaled, y_true)

if run_all_btn:
    prog = st.progress(0)
    while st.session_state.ptr < len(X_scaled):
        online_step(X_scaled, y_true)
        prog.progress(int(100 * st.session_state.ptr / len(X_scaled)))

processed = int(st.session_state.ptr)
total = int(len(X_scaled))
counts_now = st.session_state.model.counts_.astype(int)
full_clusters = int(np.sum(counts_now >= max_sizes))

# KPI cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='card'><div class='muted'>Procesadas</div><div class='big'>{processed} / {total}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='card'><div class='muted'>Clusters llenos</div><div class='big'>{full_clusters} / {int(k)}</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='card'><div class='muted'>Método</div><div class='badge'>{descriptor}</div></div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='card'><div class='muted'>Restricción</div><div class='badge'>{'Uniforme' if mode=='uniform' else 'Ground Truth'}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ========================= Tabs =========================
tab_img, tab_rubrica = st.tabs(["🖼️ Imagen por imagen", "🧪 Modo técnico (rúbrica)"])


# ========================= TAB 1: Imagen por imagen =========================
with tab_img:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("## 🔎 Imagen por imagen")
        st.caption(
            "Sube una imagen y extraemos features según el método seleccionado (Hu/HOG/SIFT/CNN). "
            "La asignación se hace por cercanía a centroides **sin entrenar** (no modifica el modelo)."
        )

        uploaded = st.file_uploader("Sube una imagen (png/jpg/webp)", type=["png", "jpg", "jpeg", "webp", "bmp"])
        img: Optional[Image.Image] = None
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

        colbtn1, colbtn2 = st.columns(2)
        predict_btn = colbtn1.button("🎯 Clasificar ", type="primary", use_container_width=True)
        add_btn = colbtn2.button("➕ Agregar al modelo", use_container_width=True)

        st.caption(" para demo, usa **Clasificar (sin entrenar)**. Para rúbrica, el entrenamiento es con +1/Todo o Inicializar.")

    with right:
        st.markdown("## ✅ Resultado")

        if st.session_state.get("mapping", None) is None:
            st.info("Tip: presiona **🚀 Inicializar modelo** para que el sistema muestre Perro/Gato/etc .")

        if (predict_btn or add_btn) and img is None:
            st.warning("Sube una imagen primero.")
        elif predict_btn or add_btn:
            # extraer features LIVE del método seleccionado
            target_dim = int(X.shape[1])  # dimensión del CSV para este método/dataset
            feat = extract_features_live(descriptor, img, target_dim=target_dim)  # (dim,)
            if feat is None:
                st.stop()

            # escalar con el scaler del dataset (MISMO método)
            scaler_live: StandardScaler = st.session_state.scaler
            try:
                x_scaled_live = scaler_live.transform(feat.reshape(1, -1))[0].astype(np.float32)
            except Exception:
                st.error(
                    "Dimensión incompatible entre features extraídas y CSV del dataset.\n\n"
                    "Esto puede pasar si el extractor live no coincide con el que generó el CSV."
                )
                st.stop()

            # 1) predicción SIN entrenar (siempre)
            cluster_id, probs, fallback, order = predict_cluster_no_update(x_scaled_live)
            name = cluster_name(cluster_id)

            # 2) si el usuario insiste en “agregar”, recién ahí entrenamos con partial_fit_one
            trained_now = False
            if add_btn:
                _lab, info = st.session_state.model.partial_fit_one(x_scaled_live)
                trained_now = True
                # NO actualizamos y_true (no hay label real); mapping se mantiene (por dataset)
                # counts sí cambian, eso es esperado para online.

            if fallback:
                st.warning(f"⚠️ Cupos llenos. Asignación: **{name}**")
            else:
                if trained_now:
                    st.success(f"✅ Agregada al modelo como: **{name}** (online update)")
                else:
                    st.success(f"✅ Clasificada como: **{name}**")

            st.markdown("### 📊 Probabilidades por cluster")
            for i, p in enumerate(probs):
                st.write(f"{cluster_name(i)}: {p*100:.1f}%")
                st.progress(float(p))

            st.markdown("### 📦 Estado de restricciones")
            df_state = pd.DataFrame(
                {
                    "Cluster": [f"C{i+1}" for i in range(int(k))],
                    "Nombre": [cluster_name(i) for i in range(int(k))],
                    "Actual": st.session_state.model.counts_.astype(int),
                    "Máximo": st.session_state.model.max_sizes.astype(int),
                    "Disponible": (st.session_state.model.max_sizes.astype(int) - st.session_state.model.counts_.astype(int)).clip(min=0),
                }
            )
            st.dataframe(df_state, use_container_width=True, height=220)

            with st.expander("Detalle técnico", expanded=False):
                st.write("Orden intento (cercano→lejano):", order)
                st.write(
                    "Nota: esto es clustering. El nombre (Perro/Gato/etc) sale del mapeo por mayoría del dataset, "
                    "no de una red supervisada."
                )

    st.caption(f"Features usadas: {data.feature_file}")


# ========================= TAB 2: Modo técnico (rúbrica) =========================
with tab_rubrica:
    st.markdown("## 🧪 Modo técnico")
    st.caption(
        "Aquí está lo evaluable: procesamiento online instancia por instancia, restricción de tamaño, métricas y comparación de métodos."
    )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### ✅ Proceso de agrupamiento (online)")
        st.caption("Cada +1 procesa una instancia: asigna al centroide más cercano con cupo y actualiza centroides.")

        if len(st.session_state.history) == 0:
            st.info("Presiona **+1** o **Todo** para ver el proceso online.")
        else:
            last = st.session_state.history[-1]
            if last["fallback"]:
                st.error("⚠️ Fallback: clusters llenos, se asignó por cercanía.")
            st.success(f"Última instancia t={last['t']} → **{cluster_name(last['cluster'])}** | orden: {last['orden_intento']}")

            with st.expander("Ver detalle de la última decisión", expanded=False):
                st.write("Distancias:", last["distancias"])
                st.write("Orden intento:", last["orden_intento"])
                st.write("Tamaño clusters:", last["tamano_clusters"])
                st.write("Cupos clusters:", last["cupo_clusters"])
                st.write("Fallback:", last["fallback"])

        st.markdown("### 📜 Historial (últimas 25 instancias)")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist.tail(25), use_container_width=True)

    with col2:
        st.markdown("### 📦 Restricción de tamaño")
        df_sizes = pd.DataFrame(
            {
                "cluster": [f"C{i+1}" for i in range(int(k))],
                "nombre": [cluster_name(i) for i in range(int(k))],
                "tamaño_actual": st.session_state.model.counts_.astype(int),
                "cupo_maximo": st.session_state.model.max_sizes.astype(int),
                "disponible": (st.session_state.model.max_sizes.astype(int) - st.session_state.model.counts_.astype(int)).clip(min=0),
            }
        )
        st.dataframe(df_sizes, use_container_width=True, height=240)

        st.markdown("### 📈 Métricas")
        if len(st.session_state.labels) < max(10, int(k) + 2):
            st.info("Procesa más instancias para habilitar métricas.")
        else:
            labels = np.array(st.session_state.labels, dtype=int)
            X_eval, y_eval, labels_eval = safe_sample(X_scaled, y_true, labels, max_n=2000)
            mets = compute_metrics(X_eval, labels_eval, y_eval)

            m1, m2, m3 = st.columns(3)
            m1.metric("Silhouette", "-" if mets["silhouette"] is None else f"{mets['silhouette']:.3f}")
            m2.metric("ARI", "-" if mets["ari"] is None else f"{mets['ari']:.3f}")
            m3.metric("NMI", "-" if mets["nmi"] is None else f"{mets['nmi']:.3f}")

            with st.expander("PCA 2D (visualización)", expanded=False):
                X2 = PCA(n_components=2, svd_solver="full").fit_transform(X_eval)
                dfp = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1], "cluster": labels_eval.astype(str)})
                st.scatter_chart(dfp, x="PC1", y="PC2", color="cluster")

        st.markdown("### 🔁 Comparar métodos (Hu vs HOG vs SIFT vs CNN)")
        limit_n = st.slider("Instancias para comparar (rápido)", 300, 3000, 1000, 100)
        if st.button("Comparar métodos ahora", use_container_width=True):
            with st.spinner("Comparando..."):
                df_cmp = compare_methods(dataset, int(k), mode, int(uniform_max), limit_n=int(limit_n))
            st.dataframe(df_cmp, use_container_width=True)

            if df_cmp["Silhouette"].notna().any():
                best = df_cmp.sort_values("Silhouette", ascending=False).iloc[0]
                st.success(f"🏆 Mejor por Silhouette: **{best['Método']}** ({best['Silhouette']})")

    with st.expander("🎤 Información extra", expanded=False):
        st.write(
            "• El descriptor (Hu/HOG/SIFT/CNN) transforma cada imagen en un vector.\n"
            "• El algoritmo online procesa instancia por instancia: distancia a centroides → asigna al más cercano con cupo → actualiza centroide.\n"
            "• Restricción de tamaño:\n"
            "  - Uniforme: mismo cupo para todos los clusters.\n"
            "  - Ground Truth: cupo = cantidad real de cada clase (solo para experimentos/evaluación).\n"
            "• Métricas: Silhouette (interna) + ARI/NMI (externas si existe columna class).\n"
            "• Los nombres Perro/Gato/etc se muestran por mapeo cluster→clase por mayoría (no es clasificación supervisada)."
        )

st.caption(f"Dataset actual: {pretty_dataset_name(dataset)} | Features: {data.feature_file}")
