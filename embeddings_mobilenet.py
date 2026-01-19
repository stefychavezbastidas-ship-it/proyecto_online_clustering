import os
import numpy as np
import pandas as pd
import tensorflow as tf

FRUITS_PATH  = os.path.join("datasets", "fruits")
ANIMALS_PATH = os.path.join("datasets", "animals")

OUT_DIR = "features_out"
os.makedirs(OUT_DIR, exist_ok=True)

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def list_images(dataset_path, dataset_name):
    rows = []
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(EXTS)]
        for img_name in imgs:
            rows.append([dataset_name, cls, os.path.join(cls_dir, img_name), img_name])
    return pd.DataFrame(rows, columns=["dataset", "class", "path", "file"])

def make_model():
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"  # -> vector fijo
    )
    return base

def load_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def compute_embeddings(df, model, batch_size=64):
    paths = df["path"].values

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    emb = model.predict(ds, verbose=1)
    emb = emb.astype(np.float32)

    # Normalización L2 (recomendado para clustering)
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norm
    return emb

def save_npz_and_meta(X, meta_df, name):
    npz_path  = os.path.join(OUT_DIR, f"X_emb_{name}.npz")
    meta_path = os.path.join(OUT_DIR, f"meta_emb_{name}.csv")

    np.savez_compressed(npz_path, X=X)
    meta_df[["dataset", "class", "file"]].to_csv(meta_path, index=False)

    print(f"\n✅ Guardado embeddings: {npz_path}  shape={X.shape}")
    print(f"✅ Guardado meta:      {meta_path} filas={len(meta_df)}\n")

def main():
    model = make_model()

    df_fruits  = list_images(FRUITS_PATH, "fruits")
    df_animals = list_images(ANIMALS_PATH, "animals")

    print("🍎 Embeddings frutas:", len(df_fruits))
    Xf = compute_embeddings(df_fruits, model, batch_size=64)
    save_npz_and_meta(Xf, df_fruits, "fruits")

    print("🐶 Embeddings animales:", len(df_animals))
    Xa = compute_embeddings(df_animals, model, batch_size=64)
    save_npz_and_meta(Xa, df_animals, "animals")

if __name__ == "__main__":
    main()
