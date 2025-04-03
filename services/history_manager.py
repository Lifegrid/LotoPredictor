import pandas as pd
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class HistoryManager:
    HISTORY_FILE = "data/euromillions_clean.csv"
    PREDICTIONS_FILE = "data/predictions.json"

    @staticmethod
    def load_history():
        try:
            df = pd.read_csv(HistoryManager.HISTORY_FILE)
            history = []
            for _, row in df.iterrows():
                boules = row[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].tolist()
                etoiles = row[['etoile_1', 'etoile_2']].tolist()
                history.append({'boules': boules, 'etoiles': etoiles})
            return history
        except Exception as e:
            logger.error(f"Erreur chargement historique: {e}")
            return []

    @staticmethod
    def get_training_data():
        history = HistoryManager.load_history()
        if not history:
            logger.warning("Aucune donnée d'entraînement trouvée.")
        return history

    @staticmethod
    def save_prediction(prediction, real_results):
        try:
            pred_record = {
                "prediction": prediction,
                "real_results": real_results
            }
            path = Path(HistoryManager.PREDICTIONS_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(pred_record)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erreur enregistrement prédiction: {e}")

    @staticmethod
    def calculate_win(prediction, result):
        """Compare une grille avec un tirage réel"""
        try:
            b = set(prediction['boules'])
            e = set(prediction['etoiles'])
            real_b = set(result['boules'])
            real_e = set(result['etoiles'])
            return {
                "bons_numeros": len(b & real_b),
                "bonnes_etoiles": len(e & real_e)
            }
        except Exception as e:
            logger.error(f"Erreur calcul gains: {e}")
            return {"bons_numeros": 0, "bonnes_etoiles": 0}

    @staticmethod
    def get_stats():
        """Retourne des stats générales"""
        try:
            history = HistoryManager.load_history()
            if not history:
                return {}

            all_boules = [n for h in history for n in h['boules']]
            all_etoiles = [e for h in history for e in h['etoiles']]
            return {
                "top_boules": pd.Series(all_boules).value_counts().head(5).to_dict(),
                "top_etoiles": pd.Series(all_etoiles).value_counts().head(2).to_dict()
            }
        except Exception as e:
            logger.error(f"Erreur statistiques: {e}")
            return {}
