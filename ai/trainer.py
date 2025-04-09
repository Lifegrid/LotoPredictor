import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

# Chargement des données
data_path = os.path.join("data", "euromillions_clean.csv")
df = pd.read_csv(data_path)

# Configuration
sequence_length = 20
features = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
X, y = [], []

# Séquences
for i in range(len(df) - sequence_length):
    seq_x = df.iloc[i:i+sequence_length][features].values
    seq_y = df.iloc[i+sequence_length][features].values
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# Normalisation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 7)).reshape(X.shape)
y_scaled = scaler.transform(y.reshape(-1, 7)).reshape(y.shape)

# Entrées séparées
X_boules = X_scaled[:, :, :5]
X_etoiles = X_scaled[:, :, 5:]

# Modèle LSTM
input_boules = tf.keras.Input(shape=(sequence_length, 5), name='boules_input')
input_etoiles = tf.keras.Input(shape=(sequence_length, 2), name='etoiles_input')

x1 = tf.keras.layers.LSTM(64)(input_boules)
x2 = tf.keras.layers.LSTM(32)(input_etoiles)

merged = tf.keras.layers.concatenate([x1, x2])
dense = tf.keras.layers.Dense(64, activation='relu')(merged)

# 🔧 Sortie corrigée : 7 neurones pour 5 boules + 2 étoiles
output = tf.keras.layers.Dense(7, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=[input_boules, input_etoiles], outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Entraînement
model.fit([X_boules, X_etoiles], y_scaled, epochs=50, batch_size=16, verbose=1)

# Sauvegarde
os.makedirs("ai/models", exist_ok=True)
model.save("ai/models/euromillions_model.keras")
