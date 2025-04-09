### Fichier 1: games/euromillions/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = os.path.join("data", "euromillions_clean.csv")

def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Fichier de données introuvable : {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            raise ValueError("Le fichier de données est vide")
        df = df.sort_values(by='date')
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

def prepare_input_sequence(sequence_length=20):
    df = load_data()
    if df.empty:
        return np.array([]), np.array([]), None

    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5',
                                            'etoile_1', 'etoile_2']].values
        target = df.iloc[i+sequence_length][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5',
                                             'etoile_1', 'etoile_2']].values
        X.append(seq)
        y.append(target)

    if not X:
        return np.array([]), np.array([]), None

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    X_shape = X.shape
    y_shape = y.shape

    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X_shape)
    y_scaled = scaler.transform(y.reshape(-1, y.shape[-1])).reshape(y_shape)

    return X_scaled, y_scaled, scaler

def get_latest_sequence(sequence_length=20):
    df = load_data()
    if df.empty or len(df) < sequence_length:
        return np.array([]), None

    latest_seq = df.iloc[-sequence_length:][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5',
                                             'etoile_1', 'etoile_2']].values

    scaler = MinMaxScaler()
    latest_scaled = scaler.fit_transform(latest_seq)

    return latest_scaled.reshape(1, sequence_length, 7), scaler
