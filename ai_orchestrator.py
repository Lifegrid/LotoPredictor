import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict
import pandas as pd

def prepare_sequences(filename: str) -> tuple:
    """Prépare les séquences pour la prédiction"""
    df = pd.read_csv(filename)
    
    # Séparation boules/étoiles
    boules = df[[f'boule_{i}' for i in range(1,6)]].values
    etoiles = df[[f'etoile_{i}' for i in range(1,3)]].values
    
    # Normalisation
    scaler_b = MinMaxScaler()
    scaler_e = MinMaxScaler()
    boules_scaled = scaler_b.fit_transform(boules)
    etoiles_scaled = scaler_e.fit_transform(etoiles)
    
    # Préparation séquences
    seq_length = 10
    X_b, X_e = [], []
    
    for i in range(seq_length, len(boules)):
        X_b.append(boules_scaled[i-seq_length:i])
        X_e.append(etoiles_scaled[i-seq_length:i])
    
    return np.array(X_b), np.array(X_e), scaler_b, scaler_e

def predict_grids() -> List[Dict[str, list]]:
    """Prédiction des prochaines grilles"""
    try:
        # Chargement données et modèle
        X_b, X_e, scaler_b, scaler_e = prepare_sequences("data/euromillions_clean.csv")
        model = load_model("models/hybrid_model.h5")
        
        # Prédiction
        boules_pred, etoiles_pred = model.predict([X_b[-5:], X_e[-5:]])
        
        # Post-traitement
        results = []
        for b_pred, e_pred in zip(boules_pred, etoiles_pred):
            # Dénormalisation
            boules = scaler_b.inverse_transform(b_pred.reshape(1, -1))[0]
            etoiles = scaler_e.inverse_transform(e_pred.reshape(1, -1))[0]
            
            # Arrondi et validation
            boules = np.clip(np.round(boules), 1, 50).astype(int)
            etoiles = np.clip(np.round(etoiles), 1, 12).astype(int)
            
            # Élimination doublons
            boules = sorted(list(set(boules)))[:5]
            etoiles = sorted(list(set(etoiles)))[:2]
            
            results.append({
                "boules": boules,
                "etoiles": etoiles,
                "type": "hybrid"
            })
        
        return results
    
    except Exception as e:
        print(f"Erreur de prédiction: {e}")
        # Fallback aléatoire en cas d'erreur
        return generate_fallback_predictions()

def generate_fallback_predictions() -> List[Dict[str, list]]:
    """Génère des prédictions aléatoires de secours"""
    return [{
        "boules": sorted(np.random.choice(range(1,51), 5, replace=False).tolist()),
        "etoiles": sorted(np.random.choice(range(1,13), 2, replace=False).tolist()),
        "type": "fallback"
    } for _ in range(5)]