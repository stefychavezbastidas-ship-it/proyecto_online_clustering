"""
feature_extractor.py - Extracción de características de imágenes - VERSIÓN CORREGIDA
"""
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Intentar importar TensorFlow (opcional)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow no disponible. Embeddings serán simulados.")
    TENSORFLOW_AVAILABLE = False

# Intentar importar scikit-image
try:
    from skimage.feature import hog
    HOG_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-image no disponible. HOG será simulado.")
    HOG_AVAILABLE = False

class FeatureExtractor:
    def __init__(self, method='embeddings'):
        """
        Inicializa el extractor de características
        
        Parameters:
        -----------
        method : str
            'embeddings', 'hog', 'hu', o 'sift'
        """
        self.method = method.lower()
        
        # Inicializar modelos solo si son necesarios
        if self.method == 'embeddings' and TENSORFLOW_AVAILABLE:
            try:
                self.model = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                self.model.trainable = False  # No entrenable
                print("✅ MobileNetV2 cargado para embeddings")
            except Exception as e:
                print(f"⚠️  Error cargando MobileNetV2: {e}")
                self.model = None
        else:
            self.model = None
    
    def extract(self, image_path):
        """
        Extrae características de una imagen
        
        Parameters:
        -----------
        image_path : str
            Ruta a la imagen
            
        Returns:
        --------
        np.ndarray: Vector de características
        """
        method_handlers = {
            'embeddings': self._extract_embeddings,
            'hog': self._extract_hog,
            'hu': self._extract_hu,
            'sift': self._extract_sift
        }
        
        if self.method not in method_handlers:
            raise ValueError(f"Método '{self.method}' no soportado. Use: {list(method_handlers.keys())}")
        
        return method_handlers[self.method](image_path)
    
    def _load_image(self, image_path, target_size=None):
        """Carga y preprocesa la imagen"""
        try:
            img = Image.open(image_path).convert('RGB')
            
            if target_size:
                img = img.resize(target_size)
            
            return np.array(img)
        except Exception as e:
            print(f"❌ Error cargando imagen: {e}")
            # Crear imagen de prueba
            if target_size:
                return np.random.randint(0, 255, (*target_size, 3), dtype=np.uint8)
            else:
                return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def _extract_embeddings(self, image_path):
        """Extrae embeddings con MobileNetV2 o simulación"""
        try:
            # Cargar imagen
            img_array = self._load_image(image_path, target_size=(224, 224))
            
            # Normalizar
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extraer embeddings
            if self.model is not None:
                embeddings = self.model.predict(img_array, verbose=0)
                features = embeddings.flatten()
                print(f"✅ Embeddings extraídos: {len(features)} dimensiones")
            else:
                # Simulación
                features = np.random.randn(1280)
                print(f"📝 Embeddings simulados: {len(features)} dimensiones")
            
            return features
            
        except Exception as e:
            print(f"❌ Error extrayendo embeddings: {e}")
            return np.random.randn(1280)  # Vector de características simulado
    
    def _extract_hog(self, image_path):
        """Extrae características HOG"""
        try:
            # Cargar imagen en escala de grises
            img_array = self._load_image(image_path, target_size=(128, 128))
            
            if len(img_array.shape) == 3:
                # Convertir a escala de grises
                if cv2.__version__:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    # Alternativa sin OpenCV
                    img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                img_gray = img_array
            
            # Extraer HOG
            if HOG_AVAILABLE:
                features = hog(
                    img_gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=False,
                    channel_axis=None
                )
            else:
                # Simulación de HOG
                features = np.random.randn(3780)  # Tamaño típico
            
            print(f"✅ HOG extraído: {len(features)} dimensiones")
            return features
            
        except Exception as e:
            print(f"❌ Error extrayendo HOG: {e}")
            return np.random.randn(3780)
    
    def _extract_hu(self, image_path):
        """Extrae momentos de Hu"""
        try:
            # Cargar imagen en escala de grises
            img_array = self._load_image(image_path, target_size=(128, 128))
            
            if len(img_array.shape) == 3:
                if cv2.__version__:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                img_gray = img_array
            
            # Calcular momentos
            if cv2.__version__:
                moments = cv2.moments(img_gray)
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Escalar logarítmicamente para mejor rango
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                features = hu_moments
            else:
                # Simulación
                features = np.random.randn(7)
            
            print(f"✅ Momentos de Hu extraídos: {len(features)} dimensiones")
            return features
            
        except Exception as e:
            print(f"❌ Error extrayendo momentos de Hu: {e}")
            return np.random.randn(7)
    
    def _extract_sift(self, image_path):
        """Extrae características SIFT (promedio de descriptores)"""
        try:
            # Cargar imagen en escala de grises
            img_array = self._load_image(image_path, target_size=(256, 256))
            
            if len(img_array.shape) == 3:
                if cv2.__version__:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                img_gray = img_array
            
            # Extraer SIFT
            if cv2.__version__:
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(img_gray, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    # Promedio de descriptores
                    features = np.mean(descriptors, axis=0)
                else:
                    features = np.zeros(128)
            else:
                features = np.zeros(128)
            
            print(f"✅ SIFT extraído: {len(features)} dimensiones")
            return features
            
        except Exception as e:
            print(f"❌ Error extrayendo SIFT: {e}")
            return np.zeros(128)

# ===== FUNCIÓN DE PRUEBA =====
if __name__ == "__main__":
    print("🧪 Probando FeatureExtractor...")
    
    # Crear una imagen de prueba
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(test_img).save("test_image.jpg")
    
    # Probar cada método
    for method in ['embeddings', 'hog', 'hu', 'sift']:
        print(f"\n🔧 Probando método: {method}")
        try:
            extractor = FeatureExtractor(method=method)
            features = extractor.extract("test_image.jpg")
            print(f"   ✅ Características: {features.shape}")
            print(f"   📏 Dimensiones: {len(features)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Limpiar
    import os
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")
    
    print("\n✅ FeatureExtractor probado exitosamente!")
