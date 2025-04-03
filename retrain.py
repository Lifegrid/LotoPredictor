from ai.predictor import EuroMillionsPredictor
from services.history_manager import HistoryManager
import numpy as np
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialisation
        logger.info("⚙️ Initialisation du prédicteur...")
        predictor = EuroMillionsPredictor()

        # Nettoyage des données
        logger.info("🧹 Nettoyage de l'historique...")
        cleaned_history = HistoryManager.clean_history()
        logger.info(f"Historique nettoyé - {len(cleaned_history)} entrées valides")

        # Vérification des données
        training_data = HistoryManager.get_training_data()
        logger.info(f"Données d'entraînement disponibles: {len(training_data)} tirages")
        
        if len(training_data) < 10:
            logger.warning("Attention: Moins de 10 tirages valides disponibles")

        # Ré-entraînement
        logger.info("🔄 Lancement du ré-entraînement...")
        success = predictor.retrain_model()
        
        if success:
            logger.info("✅ Ré-entraînement terminé avec succès!")
            
            # Test de prédiction
            try:
                prediction = predictor.predict_next()
                logger.info(f"🔮 Prédiction test - Boules: {prediction['boules']} | Étoiles: {prediction['etoiles']}")
            except Exception as e:
                logger.error(f"❌ Erreur lors du test de prédiction: {str(e)}")
        else:
            logger.error("❌ Échec du ré-entraînement")

    except Exception as e:
        logger.error(f"💥 Erreur critique: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()