import os
import pathlib
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class Config:
    # Chemins
    BASE_DIR = pathlib.Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'ai' / 'models'
    TEMPLATE_DIR = BASE_DIR / 'templates'
    
    # Fichiers
    HISTORICAL_DATA = DATA_DIR / "euromillions.csv"
    PREDICTIONS_HISTORY = DATA_DIR / "predictions.json"
    
    # Sécurité
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-key-123')
    
    # Paramètres LSTM
    LOOKBACK_WINDOW = 30  # Nombre de tirages historiques pour les prédictions
    PREDICTION_DAYS = 3   # Jours à prédire
    
    @classmethod
    def ensure_dirs(cls):
        """Crée les répertoires requis"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)