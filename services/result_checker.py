import json
import logging
from datetime import datetime
from services.history_manager import HistoryManager

logger = logging.getLogger(__name__)

class ResultChecker:

    @staticmethod
    def compare_all():
        try:
            with open("data/predictions.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Erreur lecture predictions.json : {e}")
            return

        results = HistoryManager.load_history()
        if not results:
            logger.warning("Aucun résultat réel pour comparer.")
            return

        for entry in data:
            prediction_date = entry.get("date")
            prediction = entry.get("prediction")

            if not prediction_date or not prediction:
                continue

            matching_draw = next((r for r in results if r.get("date") == prediction_date), None)
            if not matching_draw:
                continue

            gain_info = HistoryManager.calculate_win(prediction, matching_draw)
            entry["gain"] = gain_info

        try:
            with open("data/predictions.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur écriture predictions.json : {e}")
