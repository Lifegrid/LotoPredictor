import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# 📥 Chargement des données
DATA_PATH = "euromillions_clean.csv"
df = pd.read_csv(DATA_PATH)
boule_cols = ["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"]
etoile_cols = ["etoile_1", "etoile_2"]

# ✅ Préparation des données pour le LSTM
sequence_length = 10
columns = boule_cols + etoile_cols
data = df[columns].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X = []
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])
X = np.array(X)
last_sequence = np.expand_dims(X[-1], axis=0)

# 🔮 Prédiction LSTM
model = load_model("model_euromillions.keras")
lstm_prediction = model.predict(last_sequence)[0]
lstm_numbers = np.round(scaler.inverse_transform([lstm_prediction])[0]).astype(int)
boules_lstm = sorted(np.clip(lstm_numbers[:5], 1, 50))
etoiles_lstm = sorted(np.clip(lstm_numbers[5:], 1, 12))

# 📈 Analyse fréquentielle
boules = df[boule_cols].values.ravel()
etoiles = df[etoile_cols].values.ravel()
boule_freq = Counter(boules)
etoile_freq = Counter(etoiles)
top_boules = [num for num, _ in boule_freq.most_common(5)]
top_etoiles = [num for num, _ in etoile_freq.most_common(2)]

# 🧬 Génération génétique améliorée
def generate_grid():
    grid_boules = sorted(random.sample(range(1, 51), 5))
    grid_etoiles = sorted(random.sample(range(1, 13), 2))
    return grid_boules, grid_etoiles

population_size = 100
population = [generate_grid() for _ in range(population_size)]

# 🎯 Sélection basée sur la fréquence
fitness = []
for boules, etoiles in population:
    score = sum(boule_freq[b] for b in boules) + sum(etoile_freq[e] for e in etoiles)
    fitness.append((score, boules, etoiles))

fitness.sort(reverse=True)

# ✅ Affichage final combiné
print("\n🔮 Grille LSTM :", boules_lstm, "⭐", etoiles_lstm)
print("📊 Grille Fréquences :", top_boules, "⭐", top_etoiles)
print("🧬 Grilles Génétique :")
for i in range(5):
    print(f"Grille {i+1} : {fitness[i][1]} ⭐ {fitness[i][2]}")
