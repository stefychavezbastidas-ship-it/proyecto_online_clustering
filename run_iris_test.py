import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from online_constrained_kmeans import OnlineConstrainedKMeans

# Cargar Iris
iris = load_iris()
X = iris.data
y = iris.target

# Escalado
X = StandardScaler().fit_transform(X)

# Parámetros
k = 3
max_sizes = [50, 50, 50]  # Iris tiene 50 muestras por clase

# Modelo online con restricción
model = OnlineConstrainedKMeans(
    n_clusters=k,
    max_sizes=max_sizes,
    random_state=42
)

# Clustering online
pred = model.partial_fit_predict(X)

print("=== IRIS | CLUSTERING ONLINE CON RESTRICCIÓN ===")
print("Tamaños esperados:", max_sizes)
print("Tamaños obtenidos:", np.bincount(pred))

print("\nMétricas internas:")
print("Silhouette:", silhouette_score(X, pred))
print("Davies-Bouldin:", davies_bouldin_score(X, pred))

print("\nMétricas externas:")
print("ARI:", adjusted_rand_score(y, pred))
print("NMI:", normalized_mutual_info_score(y, pred))
