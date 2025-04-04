import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import random
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FDJScraper:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    BASE_URL = "https://www.fdj.fr/jeux-de-tirage/euromillions/resultats"

    @staticmethod
    def _make_request(url: str) -> Optional[BeautifulSoup]:
        try:
            response = requests.get(url, headers=FDJScraper.HEADERS, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            logger.error(f"Erreur requête HTTP: {str(e)}")
            return None

    @staticmethod
    def _validate_result(result: Dict) -> bool:
        return (isinstance(result, dict) and
                'date' in result and
                'boules' in result and len(result['boules']) == 5 and
                all(1 <= n <= 50 for n in result['boules']) and
                'etoiles' in result and len(result['etoiles']) == 2 and
                all(1 <= n <= 12 for n in result['etoiles']))

    @staticmethod
    def get_complete_history() -> List[Dict]:
        """Récupère l'historique complet des tirages."""
        history = FDJScraper._get_all_results()
        if history:
            return history
        else:
            logger.warning("Aucun historique récupéré, génération d'historique fictif.")
            return FDJScraper._generate_fake_history(50)

    @staticmethod
    def _get_all_results() -> List[Dict]:
        """Récupère l'historique complet des résultats depuis le site FDJ"""
        results = []
        page = 1
        while True:
            soup = FDJScraper._make_request(f"{FDJScraper.BASE_URL}/page-{page}")
            if not soup:
                break
            results_items = soup.find_all("div", class_="tirage-item")
            if not results_items:
                break
            for item in results_items:
                date_str = item.select_one(".tirage-date").text.strip()
                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                boules = [int(x.text) for x in item.select(".ball")[:5]]
                etoiles = [int(x.text) for x in item.select(".star-ball")[:2]]
                results.append({
                    "date": date_obj.strftime('%Y-%m-%d'),
                    "boules": sorted(boules),
                    "etoiles": sorted(etoiles)
                })
            page += 1
        return results

    @staticmethod
    def _generate_fake_history(count: int = 50) -> List[Dict]:
        """Génère un historique fictif pour les tests."""
        return [{
            "date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        } for i in range(count, 0, -1)]

    @staticmethod
    def get_next_draw_date() -> Optional[datetime]:
        """Récupère la date du prochain tirage (si visible sur le site)."""
        url = "https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/resultats"
        try:
            soup = FDJScraper._make_request(url)
            if not soup:
                return None

            tag = soup.find("p", string=lambda x: x and "Prochain tirage" in x)
            if tag:
                raw = tag.text.split(":", 1)[-1].strip().replace("à", "").strip()
                return datetime.strptime(raw, "%A %d %B %Y %Hh%M")
        except Exception as e:
            logger.warning(f"[FDJScraper] Erreur récupération prochaine date : {e}")
        return None
