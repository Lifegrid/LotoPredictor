import numpy as np
import logging
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

class EuroMillionsPredictor:
    def __init__(self):
        try:
            model_path = os.path.join("ai", "models", "euromillions_model.keras")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Modèle LSTM chargé.")
            else:
                logger.warning("Fichier modèle introuvable.")
                self.model = None
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            self.model = None

    def predict_next(self, history):
        try:
            if not self.model or not history or len(history) < 20:
                logger.warning("Pas assez de données valides pour la prédiction.")
                return None

            # Extraction des boules et étoiles valides
            sequences_boules = []
            sequences_etoiles = []

            for entry in history:
                if isinstance(entry, dict):
                    boules = entry.get('boules', [])
                    etoiles = entry.get('etoiles', [])
                    if len(boules) == 5 and len(etoiles) == 2:
                        sequences_boules.append(boules)
                        sequences_etoiles.append(etoiles)

            if len(sequences_boules) < 20 or len(sequences_etoiles) < 20:
                logger.warning("Pas assez de données valides pour la prédiction.")
                return None

            input_boules = np.array(sequences_boules[-20:]).reshape((1, 20, 5))
            input_etoiles = np.array(sequences_etoiles[-20:]).reshape((1, 20, 2))

            prediction = self.model.predict([input_boules, input_etoiles], verbose=0)

            predicted_boules = list(np.argsort(prediction[0][:50])[-5:][::-1] + 1)
            predicted_etoiles = list(np.argsort(prediction[0][50:])[-2:][::-1] + 1)

            return {
                "boules": sorted(predicted_boules),
                "etoiles": sorted(predicted_etoiles),
                "source": "lstm"
            }

        except Exception as e:
            logger.error(f"Erreur prédiction : {e}")
            return None
