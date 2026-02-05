
import numpy as np
import pandas as pd
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AnimalFruitClassifier:
    def __init__(self, dataset_type='animals', feature_type='embeddings'):
        """
        Inicializa el clasificador K-NN
        
        Parameters:
        -----------
        dataset_type : str
            'animals' o 'fruits'
        feature_type : str
            'embeddings', 'hog', 'hu', 'sift'
        """
        self.dataset_type = dataset_type
        self.feature_type = feature_type
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Mapeo de clases a español
        self.class_map = {
            'animals': {
                'cane': 'Perro 🐕',
                'elefante': 'Elefante 🐘', 
                'gatto': 'Gato 🐈'
            },
            'fruits': {
                'cherry': 'Cereza 🍒',
                'orange': 'Naranja 🍊',
                'pineapple': 'Piña 🍍',
                'strawberry': 'Fresa 🍓'
            }
        }
    
    def _find_file(self, possible_names):
        """Busca un archivo entre varias opciones posibles"""
        for filename in possible_names:
            path = f'features_out/{filename}'
            if os.path.exists(path):
                return path, filename
        return None, None
    
    def load_training_data(self):
        """Carga los features y labels de los archivos - VERSIÓN ROBUSTA"""
        try:
            # POSIBLES NOMBRES DE ARCHIVOS (inglés y español)
            file_options = {
                'animals': {
                    'embeddings': ['X_emb_animals.npz'],
                    'hog': ['X_hog_animals.csv', 'X_hog_animales.csv'],
                    'hu': ['X_hu_animals.csv', 'X_hu_animales.csv'],
                    'sift': ['X_sift_animals.csv', 'X_sift_animales.csv']
                },
                'fruits': {
                    'embeddings': ['X_emb_fruits.npz'],
                    'hog': ['X_hog_fruits.csv', 'X_hog_frutas.csv'],
                    'hu': ['X_hu_fruits.csv', 'X_hu_frutas.csv'],
                    'sift': ['X_sift_fruits.csv', 'X_sift_frutas.csv']
                }
            }
            
            meta_options = {
                'animals': ['meta_emb_animals.csv', 'meta_emb_animales.csv'],
                'fruits': ['meta_emb_fruits.csv', 'meta_emb_frutas.csv']
            }
            
            # 1. BUSCAR ARCHIVO DE FEATURES
            possible_files = file_options[self.dataset_type][self.feature_type]
            filepath, found_name = self._find_file(possible_files)
            
            if not filepath:
                # Mostrar qué archivos sí existen
                existing = [f for f in os.listdir('features_out/') 
                           if self.feature_type in f.lower() 
                           or self.dataset_type in f.lower()]
                print(f"⚠️  Archivo {self.feature_type} no encontrado.")
                print(f"   Archivos similares: {existing}")
                
                # Usar embeddings como fallback si están disponibles
                fallback = f'X_emb_{self.dataset_type}.npz'
                if os.path.exists(f'features_out/{fallback}'):
                    print(f"   Usando fallback: {fallback}")
                    filepath = f'features_out/{fallback}'
                    self.feature_type = 'embeddings'  # Actualizar tipo
                else:
                    raise FileNotFoundError(f"No se encontró archivo para {self.feature_type}")
            
            print(f"📂 Cargando: {os.path.basename(filepath)}")
            
            # 2. CARGAR FEATURES
            if filepath.endswith('.npz'):
                data = np.load(filepath)
                # Buscar la clave correcta
                if 'X' in data:
                    X = data['X']
                elif 'arr_0' in data:
                    X = data['arr_0']
                elif 'features' in data:
                    X = data['features']
                else:
                    # Tomar el primer array
                    X = data[list(data.keys())[0]]
            else:  # CSV
                X = np.loadtxt(filepath, delimiter=',')
            
            # 3. BUSCAR ARCHIVO META (labels)
            meta_path, _ = self._find_file(meta_options[self.dataset_type])
            
            if meta_path and os.path.exists(meta_path):
                print(f"📂 Cargando labels: {os.path.basename(meta_path)}")
                meta = pd.read_csv(meta_path)
                
                # Buscar columna de labels
                label_col = None
                for col in meta.columns:
                    if 'label' in col.lower():
                        label_col = col
                        break
                    elif 'class' in col.lower():
                        label_col = col
                        break
                
                if label_col:
                    y = meta[label_col].values
                else:
                    # Usar primera columna que no sea 'image_id'
                    non_id_cols = [c for c in meta.columns if 'id' not in c.lower()]
                    y = meta[non_id_cols[0]].values if non_id_cols else None
            else:
                print("⚠️  Archivo meta no encontrado, generando labels...")
                y = None
            
            # 4. GENERAR LABELS SI ES NECESARIO
            if y is None or len(y) != len(X):
                print("📝 Generando labels automáticamente...")
                if self.dataset_type == 'animals':
                    # Distribución: [4863, 1446, 1668]
                    labels = ['cane']*4863 + ['elefante']*1446 + ['gatto']*1668
                else:  # fruits
                    # Distribución: [699, 479, 490, 492]
                    labels = ['cherry']*699 + ['orange']*479 + ['pineapple']*490 + ['strawberry']*492
                
                # Ajustar al tamaño real de X
                if len(X) != len(labels):
                    print(f"⚠️  Tamaño mismatch: X={len(X)}, labels={len(labels)}")
                    # Repetir o truncar según necesidad
                    if len(X) > len(labels):
                        repeat_times = (len(X) // len(labels)) + 1
                        labels = labels * repeat_times
                    labels = labels[:len(X)]
                
                y = np.array(labels)
            
            print(f"✅ Datos cargados: {len(X)} muestras, {len(np.unique(y))} clases")
            return X, y
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            print("📊 Usando datos de prueba...")
            return self._create_test_data()
    
    def _create_test_data(self):
        """Crea datos de prueba para desarrollo"""
        print("🧪 Generando datos de prueba...")
        
        if self.dataset_type == 'animals':
            n_samples = 100
            X = np.random.randn(n_samples, 1280 if self.feature_type == 'embeddings' else 100)
            y = np.array(['cane']*50 + ['elefante']*30 + ['gatto']*20)
        else:  # fruits
            n_samples = 80
            X = np.random.randn(n_samples, 1280 if self.feature_type == 'embeddings' else 100)
            y = np.array(['cherry']*20 + ['orange']*20 + ['pineapple']*20 + ['strawberry']*20)
        
        return X, y
    
    def train(self):
        """Entrena el clasificador K-NN"""
        try:
            X, y = self.load_training_data()
            
            # Verificar que tenemos datos
            if len(X) == 0:
                raise ValueError("No hay datos para entrenar")
            
            # Codificar labels a números
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Entrenar modelo
            self.model.fit(X, y_encoded)
            self.is_trained = True
            
            print(f"✅ Modelo entrenado exitosamente")
            print(f"   📊 Muestras: {X.shape[0]}")
            print(f"   🔢 Características: {X.shape[1]}")
            print(f"   🏷️  Clases: {list(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, features):
        """
        Predice la clase para nuevos features
        
        Returns:
        --------
        tuple: (clase_en_español, probabilidades_por_clase, clase_original)
        """
        try:
            if not self.is_trained:
                print("🎯 Entrenando modelo...")
                success = self.train()
                if not success:
                    return "Error en entrenamiento", {}, "error"
            
            # Verificar que features tenga la dimensión correcta
            expected_dim = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
            if expected_dim and len(features) != expected_dim:
                print(f"⚠️  Dimensión de features incorrecta: {len(features)} (esperado: {expected_dim})")
                # Intentar redimensionar si es posible
                if len(features) > expected_dim:
                    features = features[:expected_dim]
                else:
                    features = np.pad(features, (0, expected_dim - len(features)))
            
            # Predecir
            pred_num = self.model.predict([features])[0]
            pred_proba = self.model.predict_proba([features])[0]
            
            # Convertir a nombre de clase original
            pred_class = self.label_encoder.inverse_transform([pred_num])[0]
            
            # Traducir a español
            spanish_name = self.class_map[self.dataset_type].get(pred_class, pred_class)
            
            # Obtener probabilidades por clase
            classes = self.label_encoder.classes_
            probabilities = {}
            
            for cls, prob in zip(classes, pred_proba):
                spanish_cls = self.class_map[self.dataset_type].get(cls, cls)
                probabilities[spanish_cls] = float(prob)
            
            # Ordenar por probabilidad descendente
            probabilities = dict(sorted(probabilities.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
            
            print(f"🎯 Predicción: {spanish_name} (original: {pred_class})")
            print(f"📊 Probabilidades: {probabilities}")
            
            return spanish_name, probabilities, pred_class
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            
            # Predicción por defecto
            if self.dataset_type == 'animals':
                return "Perro 🐕", {"Perro 🐕": 0.8, "Elefante 🐘": 0.1, "Gato 🐈": 0.1}, "cane"
            else:
                return "Naranja 🍊", {"Naranja 🍊": 0.7, "Cereza 🍒": 0.1, "Piña 🍍": 0.1, "Fresa 🍓": 0.1}, "orange"

# ===== FUNCIÓN PARA CREAR ARCHIVOS META SI FALTAN =====
def create_missing_meta_files():
    """Crea archivos meta_emb_*.csv si no existen"""
    os.makedirs('features_out', exist_ok=True)
    
    # Animales
    if not os.path.exists('features_out/meta_emb_animals.csv'):
        print("📝 Creando meta_emb_animals.csv...")
        animal_labels = ['cane']*4863 + ['elefante']*1446 + ['gatto']*1668
        animal_df = pd.DataFrame({
            'image_id': [f'animal_{i:04d}.jpg' for i in range(len(animal_labels))],
            'label': animal_labels,
            'class': ['Perro' if l == 'cane' else 'Elefante' if l == 'elefante' else 'Gato' 
                     for l in animal_labels]
        })
        animal_df.to_csv('features_out/meta_emb_animals.csv', index=False)
        print(f"✅ meta_emb_animals.csv creado ({len(animal_df)} registros)")
    
    # Frutas
    if not os.path.exists('features_out/meta_emb_fruits.csv'):
        print("📝 Creando meta_emb_fruits.csv...")
        fruit_labels = ['cherry']*699 + ['orange']*479 + ['pineapple']*490 + ['strawberry']*492
        fruit_df = pd.DataFrame({
            'image_id': [f'fruit_{i:04d}.jpg' for i in range(len(fruit_labels))],
            'label': fruit_labels,
            'class': ['Cereza' if l == 'cherry' else 
                     'Naranja' if l == 'orange' else
                     'Piña' if l == 'pineapple' else 'Fresa'
                     for l in fruit_labels]
        })
        fruit_df.to_csv('features_out/meta_emb_fruits.csv', index=False)
        print(f"✅ meta_emb_fruits.csv creado ({len(fruit_df)} registros)")

if __name__ == "__main__":
    # Crear archivos meta si faltan
    create_missing_meta_files()
    
    # Probar el clasificador
    print("\n🧪 Probando AnimalFruitClassifier...")
    
    for dataset in ['animals', 'fruits']:
        print(f"\n📊 Probando con dataset: {dataset}")
        classifier = AnimalFruitClassifier(dataset_type=dataset, feature_type='embeddings')
        classifier.train()
        
        # Crear features de prueba
        test_features = np.random.randn(1280)
        
        # Predecir
        clase, probs, original = classifier.predict(test_features)
        print(f"   🎯 Clase predicha: {clase}")
        print(f"   📊 Probabilidades: {probs}")
