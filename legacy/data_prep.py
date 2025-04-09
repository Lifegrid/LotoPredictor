import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 📦 Fonction pour charger, scaler et transformer les données en séquences
def prepare_sequences(csv_path, sequence_length=10):
    df = pd.read_csv(csv_path)
    df = df.sort_values("date")

    # Garder seulement les colonnes numériques (boules + étoiles)
    columns = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
    data = df[columns].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])

    X, y = np.array(X), np.array(y)
    return X, y, scaler
