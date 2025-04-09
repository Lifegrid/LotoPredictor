from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

class FDJScraper:
    @staticmethod
    def get_last_results():
        logger.warning("Scraper désactivé. Résultats fictifs fournis.")
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        }

    @staticmethod
    def get_next_draw_date():
        from datetime import timedelta
        now = datetime.now()
        weekday = now.weekday()
        days_until_draw = 1 if weekday in [0,3] else (3 if weekday == 1 else (4 if weekday == 4 else (1 - weekday)%7))
        next_draw = now + timedelta(days=days_until_draw)
        return next_draw.replace(hour=20, minute=30, second=0, microsecond=0)
