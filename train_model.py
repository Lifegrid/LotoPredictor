import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import os

# Création du dossier models si inexistant
os.makedirs("models", exist_ok=True)

# Chargement des données
df = pd.read_csv("euromillions_clean.csv")

# Séparation boules/étoiles
boules = df[[f'boule_{i}' for i in range(1,6)]].values
etoiles = df[[f'etoile_{i}' for i in range(1,3)]].values

# Normalisation séparée
scaler_boules = MinMaxScaler()
scaler_etoiles = MinMaxScaler()
boules_scaled = scaler_boules.fit_transform(boules)
etoiles_scaled = scaler_etoiles.fit_transform(etoiles)

# Préparation des séquences
sequence_length = 10
X_boules, X_etoiles, y_boules, y_etoiles = [], [], [], []

for i in range(sequence_length, len(boules)):
    X_boules.append(boules_scaled[i-sequence_length:i])
    X_etoiles.append(etoiles_scaled[i-sequence_length:i])
    y_boules.append(boules_scaled[i])
    y_etoiles.append(etoiles_scaled[i])

X_boules = np.array(X_boules)
X_etoiles = np.array(X_etoiles)
y_boules = np.array(y_boules)
y_etoiles = np.array(y_etoiles)

# Création du modèle hybride
def create_model():
    # Inputs
    input_boules = Input(shape=(sequence_length, 5))
    input_etoiles = Input(shape=(sequence_length, 2))
    
    # Couches LSTM
    lstm_b = LSTM(64, return_sequences=True)(input_boules)
    lstm_b = BatchNormalization()(lstm_b)
    lstm_b = LSTM(32)(lstm_b)
    
    lstm_e = LSTM(32)(input_etoiles)
    
    # Fusion
    merged = Concatenate()([lstm_b, lstm_e])
    dense = Dense(64, activation='swish')(merged)
    
    # Outputs
    out_boules = Dense(5, activation='sigmoid')(dense)
    out_etoiles = Dense(2, activation='sigmoid')(dense)
    
    model = Model(inputs=[input_boules, input_etoiles], outputs=[out_boules, out_etoiles])
    model.compile(optimizer='adam', loss='mse')
    return model

# Entraînement
model = create_model()
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    [X_boules, X_etoiles],
    [y_boules, y_etoiles],
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Sauvegarde
model.save("models/hybrid_model.h5")
print("✅ Modèle entraîné et sauvegardé")