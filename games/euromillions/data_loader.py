import logging
from services.history_manager import HistoryManager

logger = logging.getLogger(__name__)

def load_training_data():
    try:
        history = HistoryManager.get_training_data()
        if isinstance(history, list) and len(history) > 0:
            return history
        else:
            raise ValueError("Historique vide ou invalide")
    except Exception as e:
        logger.error(f"Erreur chargement données d'entraînement : {e}")
        return []
