from collections import Counter
import pandas as pd
from config import Config

class PatternAnalyzer:
    def __init__(self):
        self.df = pd.read_csv(Config.HISTORICAL_DATA)
        
    def get_frequencies(self):
        numbers = pd.concat([
            self.df['boule_1'], self.df['boule_2'],
            self.df['boule_3'], self.df['boule_4'],
            self.df['boule_5']
        ])
        return {
            'hot_numbers': Counter(numbers).most_common(10),
            'cold_numbers': Counter(numbers).most_common()[-10:],
            'patterns': self.find_common_patterns()
        }
    
    def find_common_patterns(self):
        # Analyse des combinaisons gagnantes
        pass