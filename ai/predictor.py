import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from datetime import datetime
import joblib
import random
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EuroMillionsPredictor:
    def __init__(self):
        self.model_dir = Path(__file__).parent / "models"
        self.model_path = self.model_dir / "euromillions_model.keras"
        self.scaler_b_path = self.model_dir / "scaler_boules.pkl"
        self.scaler_e_path = self.model_dir / "scaler_etoiles.pkl"
        
        self.model = None
        self.scaler_boules = None
        self.scaler_etoiles = None

        self.initialize_model()

    def _validate_history(self, history):
        if not isinstance(history, list):
            return False
        for entry in history:
            if not isinstance(entry, dict):
                return False
            if 'boules' not in entry or 'etoiles' not in entry:
                return False
            if not isinstance(entry['boules'], list) or len(entry['boules']) != 5:
                return False
            if not isinstance(entry['etoiles'], list) or len(entry['etoiles']) != 2:
                return False
        return True

    def initialize_model(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            if self.model_path.exists():
                try:
                    self.model = load_model(self.model_path)
                    self.scaler_boules = joblib.load(self.scaler_b_path)
                    self.scaler_etoiles = joblib.load(self.scaler_e_path)
                    logger.info("Modèle existant chargé avec succès")
                except Exception as e:
                    logger.error(f"Erreur chargement modèle: {e}")
                    self.train_model()
            else:
                logger.info("Aucun modèle trouvé - Lancement de l'entraînement")
                self.train_model()
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise

    def train_model(self, history=None):
        try:
            from services.history_manager import HistoryManager

            if history is None:
                history = HistoryManager.get_training_data()
                if not history or not self._validate_history(history):
                    logger.warning("Génération de données fictives")
                    history = [{
                        "boules": sorted(random.sample(range(1, 51), 5)),
                        "etoiles": sorted(random.sample(range(1, 13), 2))
                    } for _ in range(50)]

            if len(history) < 10:
                raise ValueError("Historique insuffisant (minimum 10 tirages)")

            boules = np.array([x['boules'] for x in history])
            etoiles = np.array([x['etoiles'] for x in history])

            self.scaler_boules = MinMaxScaler().fit(boules)
            self.scaler_etoiles = MinMaxScaler().fit(etoiles)

            boules_scaled = self.scaler_boules.transform(boules)
            etoiles_scaled = self.scaler_etoiles.transform(etoiles)

            seq_length = 10
            X_b, X_e, y_b, y_e = [], [], [], []

            for i in range(seq_length, len(boules)):
                X_b.append(boules_scaled[i - seq_length:i])
                X_e.append(etoiles_scaled[i - seq_length:i])
                y_b.append(boules_scaled[i])
                y_e.append(etoiles_scaled[i])

            input_b = Input(shape=(seq_length, 5), name='input_boules')
            input_e = Input(shape=(seq_length, 2), name='input_etoiles')

            lstm_b = LSTM(64, return_sequences=True)(input_b)
            lstm_b = LSTM(32)(lstm_b)
            lstm_e = LSTM(32)(input_e)

            merged = Concatenate()([lstm_b, lstm_e])
            dense = Dense(64, activation='relu')(merged)

            out_b = Dense(5, activation='sigmoid', name='boules')(dense)
            out_e = Dense(2, activation='sigmoid', name='etoiles')(dense)

            self.model = Model(inputs=[input_b, input_e], outputs=[out_b, out_e])

            self.model.compile(
                optimizer='adam',
                loss={'boules': 'mean_squared_error', 'etoiles': 'mean_squared_error'},
                metrics={'boules': ['mae', MeanSquaredError()], 'etoiles': ['mae', MeanSquaredError()]}
            )

            self.model.fit(
                [np.array(X_b), np.array(X_e)],
                [np.array(y_b), np.array(y_e)],
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10)],
                verbose=1
            )

            self.model.save(self.model_path)
            joblib.dump(self.scaler_boules, self.scaler_b_path)
            joblib.dump(self.scaler_etoiles, self.scaler_e_path)
            logger.info("Modèle entraîné et sauvegardé")

        except Exception as e:
            logger.error(f"Erreur entraînement: {e}\n{traceback.format_exc()}")
            raise

    def retrain_model(self, new_data=None):
        logger.info("Lancement du ré-entraînement...")
        try:
            self.train_model(new_data)
            logger.info("Ré-entraînement terminé avec succès")
            return True
        except Exception as e:
            logger.error(f"Échec du ré-entraînement: {e}")
            return False

    def predict_next(self, history=None):
        try:
            from services.history_manager import HistoryManager

            if history is None:
                history = HistoryManager.get_training_data()

            if not self._validate_history(history):
                logger.error("Historique invalide")
                return self._generate_random_prediction()

            if len(history) < 10:
                logger.warning("Historique insuffisant")
                return self._generate_random_prediction()

            last_boules = np.array([x['boules'] for x in history[-10:]])
            last_etoiles = np.array([x['etoiles'] for x in history[-10:]])

            last_boules = self.scaler_boules.transform(last_boules)
            last_etoiles = self.scaler_etoiles.transform(last_etoiles)

            pred_b, pred_e = self.model.predict(
                [last_boules[np.newaxis, ...], last_etoiles[np.newaxis, ...]],
                verbose=0
            )

            pred_b = pred_b.reshape(1, -1)
            pred_e = pred_e.reshape(1, -1)

            boules = self._process_prediction(self.scaler_boules.inverse_transform(pred_b)[0], 1, 50)
            etoiles = self._process_prediction(self.scaler_etoiles.inverse_transform(pred_e)[0], 1, 12)

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "boules": sorted(boules),
                "etoiles": sorted(etoiles),
                "type": "LSTM"
            }

        except Exception as e:
            logger.error(f"Erreur prédiction: {e}\n{traceback.format_exc()}")
            return self._generate_random_prediction()

    def _process_prediction(self, prediction, min_val, max_val):
        numbers = np.clip(np.round(prediction), min_val, max_val).astype(int)
        unique = list(set(numbers))
        target_length = 5 if max_val == 50 else 2
        while len(unique) < target_length:
            n = random.randint(min_val, max_val)
            if n not in unique:
                unique.append(n)
        return sorted(unique[:target_length])

    def _generate_random_prediction(self):
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2)),
            "type": "Aléatoire"
        }
