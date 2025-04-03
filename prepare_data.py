import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def prepare_sequences(filename: str, sequence_length: int = 10) -> Tuple[np.array, np.array, MinMaxScaler, MinMaxScaler]:
    """Prépare les séquences pour l'entraînement"""
    df = pd.read_csv(filename)
    
    # Extraction des données
    boules = df[[f"boule_{i}" for i in range(1,6)]].values
    etoiles = df[[f"etoile_{i}" for i in range(1,3)]].values
    
    # Normalisation
    scaler_b = MinMaxScaler(feature_range=(0,1))
    scaler_e = MinMaxScaler(feature_range=(0,1))
    boules_norm = scaler_b.fit_transform(boules)
    etoiles_norm = scaler_e.fit_transform(etoiles)
    
    # Création des séquences
    X, y = [], []
    for i in range(sequence_length, len(boules)):
        seq_b = boules_norm[i-sequence_length:i]
        seq_e = etoiles_norm[i-sequence_length:i]
        X.append(np.concatenate([seq_b, seq_e], axis=1))
        y.append(np.concatenate([boules[i], etoiles[i]]))
    
    return np.array(X), np.array(y), scaler_b, scaler_e