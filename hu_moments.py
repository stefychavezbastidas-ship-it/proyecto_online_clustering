import os
import cv2
import numpy as np
import pandas as pd

# =========================
# RUTAS (ajusta si tu carpeta se llama distinto)
# =========================
FRUITS_PATH  = os.path.join("datasets", "fruits")
ANIMALS_PATH = os.path.join("datasets", "animals")

OUT_DIR = "features_out"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# PREPROCESAMIENTO para Hu
# =========================
def preprocess_for_hu(img_bgr, size=(128, 128)):
    img_resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

# =========================
# HU MOMENTS (7 features)
# =========================
def hu_moments_features(binary_img):
    m = cv2.moments(binary_img)
    hu = cv2.HuMoments(m).flatten()
    eps = 1e-12
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log

# =========================
# Construir dataset Hu
# =========================
def build_hu_dataset(dataset_path, dataset_name):
    rows = []
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ No se pudo leer: {img_path}")
                continue

            bin_img = preprocess_for_hu(img)
            hu = hu_moments_features(bin_img)

            row = {
                "dataset": dataset_name,
                "class": cls,
                "file": img_name
            }
            for i in range(7):
                row[f"hu_{i+1}"] = hu[i]

            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"X_hu_{dataset_name}.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Dataset: {dataset_name}")
    print(f"✅ Guardado: {out_csv}")
    print(f"✅ Filas (imágenes): {len(df)}")
    print(f"✅ Clases: {df['class'].nunique()}\n")
    return df

def main():
    # Verifica que existan carpetas
    if not os.path.isdir(FRUITS_PATH):
        raise FileNotFoundError(f"No existe: {FRUITS_PATH}")
    if not os.path.isdir(ANIMALS_PATH):
        raise FileNotFoundError(f"No existe: {ANIMALS_PATH}")

    build_hu_dataset(FRUITS_PATH, "fruits")
    build_hu_dataset(ANIMALS_PATH, "animals")

if __name__ == "__main__":
    main()
