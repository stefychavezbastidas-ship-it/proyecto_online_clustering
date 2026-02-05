import numpy as np
from online_constrained_kmeans import OnlineConstrainedKMeans

class ClusteringGUI:
    def __init__(self, k=3, constraints=[50,50,50], method='hog'):
        self.k = k
        self.constraints = constraints
        self.method = method
        self.clusterer = OnlineConstrainedKMeans(k=k, constraint_sizes=constraints)
        self.extractor = FeatureExtractor(method=method)
        self.history = []
    
    def add_image(self, image_path):
        # Extraer características
        features = self.extractor.extract(image_path)
        
        # Clasificar online
        cluster_idx = self.clusterer.partial_fit([features])
        
        # Guardar historial
        self.history.append({
            'image': image_path,
            'cluster': cluster_idx[0],
            'features': features
        })
        
        return cluster_idx[0]
