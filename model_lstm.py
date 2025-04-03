# 📁 predict_lstm.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_prep import prepare_sequences

# Chargement des données (mêmes séquences utilisées à l'entraînement)
X, y, scaler = prepare_sequences("euromillions_clean.csv", sequence_length=10)

# Chargement du modèle entraîné
model = load_model("model_euromillions.keras")

# Dernière séquence connue pour prédire le tirage suivant
last_sequence = X[-1]
last_sequence = np.expand_dims(last_sequence, axis=0)

# Nombre de grilles à prédire
n_predictions = 5
predictions = []

for _ in range(n_predictions):
    pred = model.predict(last_sequence)[0]
    pred_inverse = scaler.inverse_transform([pred])[0]
    
    # Conversion en entiers, tri et découpe
    numbers = np.round(pred_inverse[:5]).astype(int)
    stars = np.round(pred_inverse[5:]).astype(int)

    # Nettoyage : valeurs valides
    numbers = np.clip(numbers, 1, 50)
    stars = np.clip(stars, 1, 12)
    
    # Tri des boules
    numbers.sort()
    stars.sort()

    predictions.append({
        "boules": numbers.tolist(),
        "étoiles": stars.tolist()
    })

# Affichage
for i, p in enumerate(predictions, 1):
    print(f"🔮 Grille {i} : {p['boules']} ⭐ {p['étoiles']}")
