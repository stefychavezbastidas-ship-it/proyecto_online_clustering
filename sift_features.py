import os
import cv2
import numpy as np
import pandas as pd

FRUITS_PATH  = os.path.join("datasets", "fruits")
ANIMALS_PATH = os.path.join("datasets", "animals")

OUT_DIR = "features_out"
os.makedirs(OUT_DIR, exist_ok=True)

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def preprocess_for_sift(img_bgr, size=(256, 256)):
    img_resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return gray

def sift_fixed_256(gray, sift):
    kps, desc = sift.detectAndCompute(gray, None)

    # Si no hay descriptores, devuelve ceros
    if desc is None or len(desc) == 0:
        return np.zeros(256, dtype=np.float32)

    mean = desc.mean(axis=0)  # 128
    std  = desc.std(axis=0)   # 128
    feat = np.concatenate([mean, std]).astype(np.float32)  # 256
    return feat

def build_sift_csv(dataset_path, dataset_name, log_every=200):
    rows = []
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])

    sift = cv2.SIFT_create()
    total = 0

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(EXTS)]

        print(f"\n📂 Clase: {cls} | Imágenes: {len(images)}")

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                gray = preprocess_for_sift(img)
                feat = sift_fixed_256(gray, sift)

                row = {"dataset": dataset_name, "class": cls, "file": img_name}
                for i, v in enumerate(feat):
                    row[f"sift_{i+1}"] = float(v)

                rows.append(row)
                total += 1

                if idx % log_every == 0:
                    print(f"  ✅ Progreso {cls}: {idx}/{len(images)} | Total: {total}")

            except Exception as e:
                print(f"❌ Error con {img_path}: {e}")
                continue

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"X_sift_{dataset_name}.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Dataset: {dataset_name}")
    print(f"✅ Guardado: {out_csv}")
    print(f"✅ Filas: {len(df)} | Clases: {df['class'].nunique()} | Dimensión: 256\n")
    return df

def main():
    build_sift_csv(FRUITS_PATH, "fruits", log_every=200)
    build_sift_csv(ANIMALS_PATH, "animals", log_every=500)

if __name__ == "__main__":
    main()
