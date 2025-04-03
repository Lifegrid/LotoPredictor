import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from datetime import datetime
from pathlib import Path
import random
from collections import Counter
import warnings
import traceback
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import requests
from bs4 import BeautifulSoup

# Configuration de base
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-123')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Désactivation des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    logger.error(f"Erreur configuration GPU: {e}")

# Importations relatives
try:
    from ai.predictor import EuroMillionsPredictor
    from ai.genetic import GeneticOptimizer
    from services.fdj_scraper import FDJScraper
    from services.history_manager import HistoryManager
    
    predictor = EuroMillionsPredictor()
    genetic_optimizer = GeneticOptimizer()
except ImportError as e:
    logger.error(f"Erreur importation modules: {e}")
    raise

# Planificateur pour les mises à jour automatiques
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    func=predictor.retrain_model,
    trigger='cron',
    day_of_week='tue,fri',
    hour=22,
    misfire_grace_time=3600
)
scheduler.start()

def generate_frequency_plot(history=None):
    try:
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if history and len(history) > 0:
            all_numbers = []
            all_stars = []
            
            for entry in history:
                if isinstance(entry, dict) and 'results' in entry and isinstance(entry['results'], dict):
                    boules = entry['results'].get('boules', [])
                    etoiles = entry['results'].get('etoiles', [])
                    
                    if len(boules) == 5:
                        all_numbers.extend([int(n) for n in boules if 1 <= n <= 50])
                    if len(etoiles) == 2:
                        all_stars.extend([int(n) for n in etoiles if 1 <= n <= 12])

            if all_numbers and all_stars:
                num_counts = Counter(all_numbers)
                nums, counts = zip(*sorted(num_counts.items()))
                ax1.bar(nums, counts, color='#4e79a7')
                ax1.set_title('Fréquence des numéros (1-50)')
                ax1.set_xlabel('Numéro')
                ax1.set_ylabel('Fréquence')
                ax1.set_xticks(range(1, 51, 5))
                ax1.grid(axis='y', linestyle='--', alpha=0.7)
                
                star_counts = Counter(all_stars)
                stars, scounts = zip(*sorted(star_counts.items()))
                ax2.bar(stars, scounts, color='#f28e2b')
                ax2.set_title('Fréquence des étoiles (1-12)')
                ax2.set_xlabel('Étoile')
                ax2.set_ylabel('Fréquence')
                ax2.set_xticks(range(1, 13))
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Erreur génération graphique: {e}\n{traceback.format_exc()}")
        return ""

@app.route('/')
def index():
    try:
        history = HistoryManager.load_history() or []
        stats = HistoryManager.get_stats() or {}
        next_draw = FDJScraper.get_next_draw_date() or datetime.now()
        
        return render_template('index.html',
                               history=history[-5:],
                               stats=stats,
                               next_draw=next_draw.strftime("%A %d %B %Y à %Hh%M"),
                               plot_url=generate_frequency_plot(history))

    except Exception as e:
        logger.error(f"Erreur route / : {e}\n{traceback.format_exc()}")
        return render_template('error.html', message="Erreur lors du chargement de la page"), 500

def obtenir_resultats_euromillions():
    url = "https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/resultats"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        boules = [int(x.text) for x in soup.find_all("span", class_="ball-euromillions")]
        etoiles = [int(x.text) for x in soup.find_all("span", class_="star-euromillions")]

        return {"boules": boules[:5], "etoiles": etoiles[:2]}

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de requête HTTP : {e}")
        return None
    except ValueError as e:
        logger.error(f"Erreur de conversion de données : {e}")
        return None
    except Exception as e:
        logger.error(f"Une erreur inattendue s'est produite : {e}")
        return None

def convertir_historique(historique_actuel):
    historique_correct = []
    for entree in historique_actuel:
        if 'results' in entree and 'boules' in entree['results'] and 'etoiles' in entree['results']:
            historique_correct.append({
                'boules': entree['results']['boules'],
                'etoiles': entree['results']['etoiles']
            })
    return historique_correct

@app.route('/generate', methods=['POST'])
def generate_predictions():
    try:
        last_real_results = obtenir_resultats_euromillions()
        if not last_real_results or not isinstance(last_real_results, dict):
            logger.warning("Impossible de récupérer les derniers résultats réels, utilisation de données aléatoires.")
            last_real_results = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "boules": sorted(random.sample(range(1, 51), 5)),
                "etoiles": sorted(random.sample(range(1, 13), 2))
            }

        full_history = HistoryManager.load_history() or []
        historique_converti = convertir_historique(full_history)

        lstm_grids = []
        genetic_grids = []

        for _ in range(5):
            lstm_pred = predictor.predict_next(historique_converti)
            if lstm_pred and isinstance(lstm_pred, dict):
                win_info = HistoryManager.calculate_win(lstm_pred, last_real_results)
                win_info = {k: int(v) if isinstance(v, np.int64) else v for k, v in win_info.items()}
                lstm_grids.append({
                    **lstm_pred,
                    "type": "LSTM",
                    **win_info
                })

            genetic_pred = genetic_optimizer.generate_optimized_grid(historique_converti)
            if genetic_pred and isinstance(genetic_pred, dict):
                win_info = HistoryManager.calculate_win(genetic_pred, last_real_results)
                win_info = {k: int(v) if isinstance(v, np.int64) else v for k, v in win_info.items()}
                genetic_grids.append({
                    **genetic_pred,
                    "type": "Génétique",
                    **win_info
                })

        for grid in lstm_grids + genetic_grids:
            if isinstance(grid, dict):
                HistoryManager.save_prediction(grid, last_real_results)

        return render_template('index.html',
                               lstm_grids=lstm_grids,
                               genetic_grids=genetic_grids,
                               last_results=last_real_results,
                               plot_url=generate_frequency_plot(full_history),
                               stats=HistoryManager.get_stats())

    except Exception as e:
        logger.error(f"Erreur génération prédictions: {e}\n{traceback.format_exc()}")
        return render_template('error.html', message="Erreur lors de la génération des prédictions"), 500

@app.route('/update')
def update_data():
    try:
        new_data = FDJScraper.get_complete_history()
        if new_data and isinstance(new_data, list) and predictor.retrain_model(new_data):
            return redirect(url_for('index'))
        raise Exception("Données d'entraînement invalides")
    except Exception as e:
        logger.error(f"Erreur mise à jour: {e}\n{traceback.format_exc()}")
        return render_template('error.html', message=f"Erreur lors de la mise à jour: {str(e)}"), 500

@app.route('/stats')
def stats():
    try:
        history = HistoryManager.load_history() or []
        hot_numbers = predictor.get_hot_numbers() if hasattr(predictor, 'get_hot_numbers') else None
        cold_numbers = predictor.get_cold_numbers() if hasattr(predictor, 'get_cold_numbers') else None
        
        return render_template('stats.html',
                               plot_url=generate_frequency_plot(history),
                               hot=hot_numbers or {'boules': [], 'etoiles': []},
                               cold=cold_numbers or {'boules': [], 'etoiles': []},
                               stats=HistoryManager.get_stats())
    except Exception as e:
        logger.error(f"Erreur stats: {e}\n{traceback.format_exc()}")
        return render_template('error.html', message=f"Erreur statistiques: {str(e)}"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
