import os
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from collections import Counter
import random
import logging
import numpy as np

# Initialisation Flask
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-123')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configuration logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU TensorFlow
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    logger.error(f"Erreur configuration GPU : {e}")

# Imports internes
from games.euromillions.predictor import EuroMillionsPredictor
from games.euromillions.genetic_optimizer import GeneticOptimizer
from services.history_manager import HistoryManager
from services.fdj_scraper import FDJScraper
from services.result_checker import ResultChecker

predictor = EuroMillionsPredictor()
genetic_optimizer = GeneticOptimizer()

def generate_frequency_plot(history=None):
    try:
        if not history:
            history = HistoryManager.load_history()
            
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        all_numbers = []
        all_stars = []
        for entry in history:
            if isinstance(entry, dict):
                if 'boules' in entry:
                    all_numbers.extend(entry['boules'])
                if 'etoiles' in entry:
                    all_stars.extend(entry['etoiles'])

        # Si pas assez de données, générer des données de démo
        if not all_numbers:
            all_numbers = random.choices(range(1, 51), k=100)
        if not all_stars:
            all_stars = random.choices(range(1, 13), k=40)

        num_counts = Counter(all_numbers)
        star_counts = Counter(all_stars)

        nums, counts = zip(*sorted(num_counts.items()))
        ax1.bar(nums, counts, color='#4e79a7')
        ax1.set_title("Fréquences des boules (1-50)")
        ax1.set_xlabel("Boules")
        ax1.set_ylabel("Fréquence")
        ax1.grid(True)

        stars, scounts = zip(*sorted(star_counts.items()))
        ax2.bar(stars, scounts, color='#f28e2b')
        ax2.set_title("Fréquences des étoiles (1-12)")
        ax2.set_xlabel("Étoiles")
        ax2.set_ylabel("Fréquence")
        ax2.grid(True)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Erreur génération graphique : {e}")
        return ""

# Planification réentraînement
scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

# Route principale
@app.route('/')
def index():
    try:
        history = HistoryManager.load_history()
        stats = HistoryManager.get_stats()
        
        try:
            next_draw = FDJScraper.get_next_draw_date()
            next_draw_str = next_draw.strftime("%A %d %B %Y à %Hh%M")
        except Exception as e:
            logger.error(f"Erreur obtention date tirage : {e}")
            next_draw_str = "Prochain tirage non disponible"

        return render_template("index.html",
                               history=history[-5:] if history else [],
                               stats=stats,
                               next_draw=next_draw_str,
                               plot_url=generate_frequency_plot(history),
                               datetime=datetime)
    except Exception as e:
        logger.error(f"Erreur route / : {e}")
        return render_template("error.html", message="Erreur lors du chargement de la page"), 500

@app.route('/generate', methods=['POST'])
def generate_predictions():
    try:
        last_real_results = FDJScraper.get_last_results()
        if not last_real_results:
            history = HistoryManager.load_history()
            if history:
                last_real_results = history[-1]
            else:
                last_real_results = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "boules": sorted(random.sample(range(1, 51), 5)),
                    "etoiles": sorted(random.sample(range(1, 13), 2))
                }

        history = HistoryManager.load_history()
        lstm_grids = []
        genetic_grids = []

        for _ in range(5):
            lstm_pred = predictor.predict_next(history)
            if lstm_pred:
                gain = HistoryManager.calculate_win(lstm_pred, last_real_results)
                lstm_grids.append({**lstm_pred, **gain})

            genetic_pred = genetic_optimizer.generate_optimized_grid(history)
            if genetic_pred:
                gain = HistoryManager.calculate_win(genetic_pred, last_real_results)
                genetic_grids.append({**genetic_pred, **gain})

        if not lstm_grids and not genetic_grids:
            for _ in range(5):
                lstm_grids.append({
                    "boules": sorted(random.sample(range(1, 51), 5)),
                    "etoiles": sorted(random.sample(range(1, 13), 2)),
                    "source": "random",
                    "gain": "N/A"
                })
                genetic_grids.append({
                    "boules": sorted(random.sample(range(1, 51), 5)),
                    "etoiles": sorted(random.sample(range(1, 13), 2)),
                    "source": "random",
                    "gain": "N/A"
                })

        return render_template("index.html",
                               last_results=last_real_results,
                               lstm_grids=lstm_grids,
                               genetic_grids=genetic_grids,
                               stats=HistoryManager.get_stats(),
                               plot_url=generate_frequency_plot(history),
                               datetime=datetime)
    except Exception as e:
        logger.error(f"Erreur /generate : {e}")
        return render_template("error.html", message="Erreur lors de la génération des prédictions"), 500

@app.route('/compare')
def compare_predictions():
    try:
        ResultChecker.compare_all()
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Erreur /compare : {e}")
        return render_template("error.html", message="Erreur lors de la comparaison des résultats"), 500

@app.route('/stats')
def stats():
    try:
        history = HistoryManager.load_history()
        return render_template("stats.html",
                               plot_url=generate_frequency_plot(history),
                               stats=HistoryManager.get_stats())
    except Exception as e:
        logger.error(f"Erreur /stats : {e}")
        return render_template("error.html", message="Erreur lors du chargement des statistiques"), 500

# Lancement serveur
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
