import os
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class HistoryManager:
    HISTORY_FILE = "data/euromillions_clean.csv"
    PREDICTIONS_FILE = "data/predictions.json"

    @staticmethod
    def load_history():
        """Charge l'historique depuis un fichier CSV"""
        try:
            df = pd.read_csv(HistoryManager.HISTORY_FILE)
            history = []
            for _, row in df.iterrows():
                boules = row[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].tolist()
                etoiles = row[['etoile_1', 'etoile_2']].tolist()
                history.append({
                    "results": {
                        'boules': boules,
                        'etoiles': etoiles
                    }
                })
            return history
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique: {e}")
            return []

    @staticmethod
    def get_stats():
        """Calcule les statistiques basiques depuis les prédictions"""
        try:
            if not os.path.exists(HistoryManager.PREDICTIONS_FILE):
                return {"total": 0, "wins": 0, "best": None}

            with open(HistoryManager.PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total = len(data)
            wins = sum(1 for d in data if d.get("gain", {}).get("category") != "Pas de gain")
            best = sorted(data, key=lambda x: x.get("gain", {}).get("rank", 999))[:1]

            return {
                "total": total,
                "wins": wins,
                "best": best[0] if best else None
            }

        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {e}")
            return {"total": 0, "wins": 0, "best": None}

    @staticmethod
    def calculate_win(prediction, real_results):
        """Compare une prédiction avec les résultats pour détecter un gain"""
        try:
            pred_boules = set(prediction['boules'])
            pred_etoiles = set(prediction['etoiles'])
            real_boules = set(real_results['boules'])
            real_etoiles = set(real_results['etoiles'])

            boule_match = len(pred_boules & real_boules)
            etoile_match = len(pred_etoiles & real_etoiles)

            category = "Pas de gain"
            prize = "0 €"
            rank = 999

            if boule_match == 5 and etoile_match == 2:
                category = "Jackpot"
                prize = "100M€"
                rank = 1
            elif boule_match == 5 and etoile_match == 1:
                category = "2nd rang"
                prize = "500K€"
                rank = 2
            elif boule_match == 5:
                category = "3ème rang"
                prize = "100K€"
                rank = 3
            elif boule_match == 4 and etoile_match == 2:
                category = "4ème rang"
                prize = "5K€"
                rank = 4
            elif boule_match == 4 and etoile_match == 1:
                category = "5ème rang"
                prize = "500€"
                rank = 5
            elif boule_match == 3 and etoile_match == 2:
                category = "6ème rang"
                prize = "100€"
                rank = 6
            elif boule_match == 2 and etoile_match == 2:
                category = "7ème rang"
                prize = "20€"
                rank = 7
            elif boule_match == 3 and etoile_match == 1:
                category = "8ème rang"
                prize = "15€"
                rank = 8
            elif boule_match == 3:
                category = "9ème rang"
                prize = "10€"
                rank = 9

            return {
                "match": f"{boule_match} boules & {etoile_match} étoiles",
                "category": category,
                "prize": prize,
                "rank": rank
            }

        except Exception as e:
            logger.error(f"Erreur calcul gain: {e}")
            return {"match": "Erreur", "category": "Erreur", "prize": "0 €", "rank": 999}

    @staticmethod
    def save_prediction(prediction, real_results):
        try:
            def convert(obj):
                if isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [convert(i) for i in obj]
                elif isinstance(obj, dict):
                    return {str(k): convert(v) for k, v in obj.items()}
                elif hasattr(obj, "item"):
                    return obj.item()
                else:
                    return str(obj)

            prediction = convert(prediction)
            real_results = convert(real_results)

            entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": prediction,
                "last_results": real_results,
                "gain": HistoryManager.calculate_win(prediction, real_results)
            }

            file_path = HistoryManager.PREDICTIONS_FILE

            data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Fichier JSON corrompu. Réinitialisation...")
                    data = []

            data.append(entry)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erreur enregistrement prédiction: {e}")
