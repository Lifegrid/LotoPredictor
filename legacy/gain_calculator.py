import json
from typing import List, Dict

class GainCalculator:
    # Règles EuroMillions 2024
    GAIN_TABLE = {
        (5, 2): "Jackpot",
        (5, 1): "1ère Catégorie",
        (5, 0): "2ème Catégorie",
        (4, 2): "3ème Catégorie",
        (4, 1): "4ème Catégorie",
        (4, 0): "5ème Catégorie",
        (3, 2): "6ème Catégorie",
        (3, 1): "7ème Catégorie",
        (2, 2): "8ème Catégorie",
        (3, 0): "9ème Catégorie",
        (1, 2): "10ème Catégorie",
        (2, 1): "11ème Catégorie",
        (2, 0): "12ème Catégorie"
    }

    def __init__(self):
        with open('data/last_results.json') as f:
            self.last_results = json.load(f)

    def check_win(self, grid: Dict) -> Dict:
        """Compare une grille avec les résultats officiels"""
        boules_match = len(set(grid["boules"]) & set(self.last_results["boules"]))
        etoiles_match = len(set(grid["etoiles"]) & set(self.last_results["etoiles"]))
        
        gain = self.GAIN_TABLE.get(
            (boules_match, etoiles_match), 
            "Pas de gain"
        )
        
        return {
            "grid": grid,
            "result": self.last_results,
            "match": (boules_match, etoiles_match),
            "gain": gain
        }

    def save_to_history(self, prediction):
        try:
            with open('data/history.json', 'r+') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        history.append({
            "date": self.last_results["date"],
            "prediction": prediction
        })
        
        with open('data/history.json', 'w') as f:
            json.dump(history, f, indent=4)