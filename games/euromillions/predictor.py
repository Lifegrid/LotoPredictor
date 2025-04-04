import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import random
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError

from games.euromillions.data_loader import load_training_data

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

    def initialize_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.model_path.exists():
                self.model = load_model(self.model_path)
                self.scaler_boules = joblib.load(self.scaler_b_path)
                self.scaler_etoiles = joblib.load(self.scaler_e_path)
                logger.info("Modèle EuroMillions chargé avec succès")
            else:
                logger.info("Pas de modèle trouvé, lancement de l'entraînement")
                self.train_model()
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            self.train_model()

    def train_model(self, history=None):
        try:
            history = history or load_training_data()
            if len(history) < 10:
                raise ValueError("Historique insuffisant")

            boules = np.array([x['boules'] for x in history])
            etoiles = np.array([x['etoiles'] for x in history])

            self.scaler_boules = MinMaxScaler().fit(boules)
            self.scaler_etoiles = MinMaxScaler().fit(etoiles)
            boules_scaled = self.scaler_boules.transform(boules)
            etoiles_scaled = self.scaler_etoiles.transform(etoiles)

            seq_len = 10
            X_b, X_e, y_b, y_e = [], [], [], []
            for i in range(seq_len, len(boules_scaled)):
                X_b.append(boules_scaled[i-seq_len:i])
                X_e.append(etoiles_scaled[i-seq_len:i])
                y_b.append(boules_scaled[i])
                y_e.append(etoiles_scaled[i])

            input_b = Input(shape=(seq_len, 5))
            input_e = Input(shape=(seq_len, 2))
            x1 = LSTM(64, return_sequences=True)(input_b)
            x1 = LSTM(32)(x1)
            x2 = LSTM(32)(input_e)
            x = Concatenate()([x1, x2])
            x = Dense(64, activation='relu')(x)

            out_b = Dense(5, activation='sigmoid', name='boules')(x)
            out_e = Dense(2, activation='sigmoid', name='etoiles')(x)

            self.model = Model(inputs=[input_b, input_e], outputs=[out_b, out_e])
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=[MeanSquaredError()]
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

        except Exception as e:
            logger.error(f"Erreur entraînement modèle : {e}\n{traceback.format_exc()}")

    def predict_next(self, history=None):
        try:
            history = history or load_training_data()
            if len(history) < 10:
                return self._generate_random()

            X_b = np.array([x['boules'] for x in history[-10:]])
            X_e = np.array([x['etoiles'] for x in history[-10:]])
            X_b = self.scaler_boules.transform(X_b)
            X_e = self.scaler_etoiles.transform(X_e)

            pred_b, pred_e = self.model.predict(
                [X_b[np.newaxis, ...], X_e[np.newaxis, ...]], verbose=0
            )
            boules = self._process(self.scaler_boules.inverse_transform(pred_b)[0], 1, 50, 5)
            etoiles = self._process(self.scaler_etoiles.inverse_transform(pred_e)[0], 1, 12, 2)

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "boules": boules,
                "etoiles": etoiles,
                "type": "LSTM"
            }

        except Exception as e:
            logger.error(f"Erreur prédiction : {e}")
            return self._generate_random()

    def _process(self, values, min_val, max_val, count):
        values = np.clip(np.round(values), min_val, max_val).astype(int)
        unique = list(set(values))
        while len(unique) < count:
            n = random.randint(min_val, max_val)
            if n not in unique:
                unique.append(n)
        return sorted(unique[:count])

    def _generate_random(self):
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2)),
            "type": "Aléatoire"
        }
