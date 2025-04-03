import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

class EuroMillionsScraper:
    def __init__(self):
        self.url = "https://www.loterie-nationale.be/euromillions/resultats"

    def get_last_results(self):
        try:
            response = requests.get(self.url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraction des numéros (adaptez selon le site cible)
            boules = [int(num.text) for num in soup.select('.number-circle')[:5]]
            etoiles = [int(num.text) for num in soup.select('.star-circle')[:2]]
            date = datetime.now().strftime("%Y-%m-%d")
            
            return {
                "date": date,
                "boules": sorted(boules),
                "etoiles": sorted(etoiles)
            }
        except Exception as e:
            print(f"Erreur de scraping: {e}")
            return None

    def save_results(self, results):
        with open('data/last_results.json', 'w') as f:
            json.dump(results, f)