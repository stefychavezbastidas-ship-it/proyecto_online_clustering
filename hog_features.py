import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

FRUITS_PATH  = os.path.join("datasets", "fruits")
ANIMALS_PATH = os.path.join("datasets", "animals")

OUT_DIR = "features_out"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_for_hog(img_bgr, size=(128, 128)):
    img_resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def hog_vector(gray_img):
    return hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

def build_hog_dataset(dataset_path, dataset_name, log_every=200):
    rows = []
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])

    hog_dim = None
    total_done = 0

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        print(f"\n📂 Clase: {cls} | Imágenes: {len(images)}")

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(cls_dir, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ No se pudo leer: {img_path}")
                    continue

                gray = preprocess_for_hog(img)
                feat = hog_vector(gray)

                if hog_dim is None:
                    hog_dim = len(feat)
                    print(f"📌 Dimensión HOG ({dataset_name}): {hog_dim}")

                row = {"dataset": dataset_name, "class": cls, "file": img_name}
                for i, v in enumerate(feat):
                    row[f"hog_{i+1}"] = float(v)
                rows.append(row)

                total_done += 1
                if idx % log_every == 0:
                    print(f"  ✅ Progreso {cls}: {idx}/{len(images)} | Total procesadas: {total_done}")

            except Exception as e:
                print(f"❌ Error con {img_path}: {e}")
                continue

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"X_hog_{dataset_name}.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Dataset: {dataset_name}")
    print(f"✅ Guardado: {out_csv}")
    print(f"✅ Filas (imágenes): {len(df)}")
    print(f"✅ Clases: {df['class'].nunique()}\n")
    return df

def main():
    build_hog_dataset(FRUITS_PATH, "fruits", log_every=200)
    build_hog_dataset(ANIMALS_PATH, "animals", log_every=200)

if __name__ == "__main__":
    main()
