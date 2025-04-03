import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import joblib
from pathlib import Path
import keras.saving

# Configuration des chemins
MODEL_DIR = Path("ai/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lstm_model.keras"

class LSTMNetwork(Model):
    """Réseau LSTM personnalisé avec sérialisation"""
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(128, return_sequences=False)
        self.dropout = layers.Dropout(0.3)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(5)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dropout(x)
        x = self.dense1(x)
        return self.dense2(x)

    def get_config(self):
        return {}  # Configuration de base car pas de paramètres custom

    @classmethod
    def from_config(cls, config):
        return cls()

def train_and_save_model(X_train, y_train):
    """Entraîne et sauvegarde le modèle"""
    # Création du modèle
    model = LSTMNetwork()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses.MeanSquaredError(),
        metrics=['mae']
    )

    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

    # Sauvegarde
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    return history

def load_training_data():
    """Charge les données d'entraînement (à adapter)"""
    # Exemple avec des données aléatoires
    return (
        np.random.rand(100, 10, 5),  # X_train: 100 séquences de 10 tirages
        np.random.rand(100, 5)       # y_train: 100 prédictions
    )

if __name__ == '__main__':
    print("Chargement des données...")
    X_train, y_train = load_training_data()
    
    print("Début de l'entraînement...")
    history = train_and_save_model(X_train, y_train)
    
    print("Entraînement terminé avec succès!")