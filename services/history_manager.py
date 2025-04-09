import pandas as pd
import logging
import os
import json

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("data", "euromillions_clean.csv")
HISTORY_FILE = os.path.join("reports", "history.json")

class HistoryManager:
    @staticmethod
    def load_history():
        try:
            df = pd.read_csv(DATA_PATH)
            df = df.dropna()

            history = []
            for _, row in df.iterrows():
                draw = {
                    "date": row["date"],
                    "boules": sorted([int(row[f"boule_{i}"]) for i in range(1, 6)]),
                    "etoiles": sorted([int(row[f"etoile_{i}"]) for i in range(1, 3)])
                }
                history.append(draw)

            logger.info(f"Chargement de {len(history)} tirages valides.")
            return history
        except Exception as e:
            logger.error(f"Erreur chargement historique : {e}")
            return []

    @staticmethod
    def calculate_win(prediction, result):
        if not prediction or not result:
            return {"gain": "N/A", "matchs": 0}

        try:
            common_boules = len(set(prediction["boules"]) & set(result["boules"]))
            common_etoiles = len(set(prediction["etoiles"]) & set(result["etoiles"]))
            return {
                "gain": f"{common_boules} boules, {common_etoiles} étoiles",
                "matchs": common_boules + common_etoiles
            }
        except Exception as e:
            logger.error(f"Erreur calcul gain : {e}")
            return {"gain": "N/A", "matchs": 0}

    @staticmethod
    def save_prediction(prediction, result):
        try:
            if not os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "w") as f:
                    json.dump([], f)

            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)

            history.append({
                "prediction": prediction,
                "result": result,
                "timestamp": pd.Timestamp.now().isoformat()
            })

            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Erreur sauvegarde historique : {e}")

    @staticmethod
    def get_stats():
        try:
            if not os.path.exists(HISTORY_FILE):
                return {}

            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)

            combos = {}
            for entry in history:
                gain = entry.get("prediction", {}).get("gain", "N/A")
                if gain != "N/A":
                    combos[gain] = combos.get(gain, 0) + 1

            return dict(sorted(combos.items(), key=lambda item: -item[1]))
        except Exception as e:
            logger.error(f"Erreur statistiques : {e}")
            return {}
