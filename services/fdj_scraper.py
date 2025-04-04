import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FDJScraper:
    BASE_URL = "https://tirage-gagnant.com/euromillions/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    @staticmethod
    def get_last_draw_results():
        try:
            response = requests.get(FDJScraper.BASE_URL, headers=FDJScraper.HEADERS, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            date_text = soup.select_one(".tirage-date").text.strip()
            date_obj = datetime.strptime(date_text, "%d/%m/%Y")

            numbers = [int(span.text.strip()) for span in soup.select(".numeros .boule")[:5]]
            stars = [int(span.text.strip()) for span in soup.select(".etoiles .etoile")[:2]]

            return {
                "date": date_obj.strftime("%Y-%m-%d"),
                "boules": sorted(numbers),
                "etoiles": sorted(stars)
            }

        except Exception as e:
            logger.error(f"Erreur récupération résultats Euromillions : {e}")
            return None

    @staticmethod
    def get_next_draw_date():
        try:
            # Prochain tirage : mardi ou vendredi à 21h10
            today = datetime.today()
            weekday = today.weekday()
            
            if weekday < 1 or weekday == 3:
                days_until_next = 1 - weekday if weekday < 1 else 4 - weekday
            else:
                days_until_next = 7 - weekday + 1

            next_draw = today.replace(hour=21, minute=10, second=0, microsecond=0) + timedelta(days=days_until_next)
            return next_draw

        except Exception as e:
            logger.error(f"Erreur calcul prochaine date de tirage : {e}")
            return datetime.today()

    @staticmethod
    def get_complete_history():
        logger.warning("Récupération historique complet non encore implémentée dans cette version.")
        return []
