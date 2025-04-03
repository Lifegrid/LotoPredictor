import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Définir la fonction 'mse' avant de charger le modèle
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class EuroMillionsPredictor:
    def __init__(self):
        # Chemins des modèles
        self.model_dir = Path("ai/models")
        self.h5_path = self.model_dir / "hybrid_model.h5"  # Utilisation du modèle hybride
        self.keras_path = self.model_dir / "hybrid_model.keras"  # Utilisation du modèle hybride
        self.scaler_path = self.model_dir / "scaler.pkl"
        
        # Charge le modèle et le scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
    def _load_model(self):
        """Charge le modèle avec priorité au format .keras"""
        try:
            if self.keras_path.exists():
                print(f"Chargement du modèle .keras depuis {self.keras_path}")
                return models.load_model(
                    self.keras_path,
                    custom_objects={'mse': mse}  # Utilisation de 'mse' comme fonction
                )

            elif self.h5_path.exists():
                print(f"Chargement du modèle .h5 depuis {self.h5_path}")
                model = models.load_model(
                    self.h5_path,
                    custom_objects={'mse': mse}  # Utilisation de 'mse' comme fonction
                )
                model.save(self.keras_path)  # Sauvegarde du modèle au format .keras
                return model

            else:
                print("Aucun modèle trouvé - création d'un modèle minimal")
                return self._create_fallback_model()

        except Exception as e:
            print(f"Erreur de chargement du modèle: {e}")
            return self._create_fallback_model()

    def _create_fallback_model(self):
        """Crée un modèle minimal en cas d'échec"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(10, 5))
        ])
        model.compile(optimizer='adam', loss=mse)
        return model

    def _load_scaler(self):
        """Charge le scaler de normalisation"""
        try:
            if self.scaler_path.exists():
                return joblib.load(self.scaler_path)
            else:
                print("Scaler non trouvé - utilisation par défaut")
                return None
        except Exception as e:
            print(f"Erreur de chargement du scaler: {e}")
            return None

    def predict_next(self, last_results):
        """Prédit les numéros suivants"""
        try:
            print("Début de la méthode predict_next")

            # Vérification du format de last_results
            if not isinstance(last_results, list):
                raise ValueError("'last_results' doit être une liste")
            if len(last_results) < 10:
                raise ValueError("Historique insuffisant")

            # Vérification des clés 'boules' et 'etoiles' dans chaque entrée
            for i, entry in enumerate(last_results[-10:]):
                print(f"Entrée {i}: {entry}")  # Affichage de chaque entrée
                if 'boules' not in entry:
                    raise ValueError(f"L'entrée {i} ne contient pas la clé 'boules'")
                if 'etoiles' not in entry:
                    raise ValueError(f"L'entrée {i} ne contient pas la clé 'etoiles'")

            # Préparation des données
            last_numbers = np.array([r['boules'] for r in last_results[-10:]])

            if self.scaler:
                last_numbers = self.scaler.transform(last_numbers)

            # Prédiction
            prediction = self.model.predict(last_numbers[np.newaxis, ...])[0]

            # Post-traitement
            if self.scaler:
                prediction = self.scaler.inverse_transform(prediction[np.newaxis, ...])[0]

            # Processus de transformation des prédictions en numéros valides
            numbers = self._process_prediction(prediction)
            stars = sorted(np.random.choice(range(1, 13), 2, replace=False))

            # Affichage des résultats pour le débogage
            print("Prédiction des boules:", numbers)
            print("Prédiction des étoiles:", stars)

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "boules": sorted(numbers),
                "etoiles": stars
            }

        except Exception as e:
            print(f"Erreur de prédiction: {e}")
            return self._generate_random_prediction()

    def _process_prediction(self, prediction):
        """Transforme la prédiction en numéros valides"""
        numbers = np.clip(np.round(prediction), 1, 50).astype(int)
        unique = list(set(numbers))

        while len(unique) < 5:
            n = np.random.randint(1, 51)
            if n not in unique:
                unique.append(n)

        return unique[:5]

    def _generate_random_prediction(self):
        """Génère une prédiction aléatoire de secours"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "boules": sorted(np.random.choice(range(1, 51), 5, replace=False)),
            "etoiles": sorted(np.random.choice(range(1, 13), 2, replace=False))
        }
