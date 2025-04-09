import os
import shutil
from datetime import datetime

# Fichiers considérés comme obsolètes ou à archiver
FILES_TO_ARCHIVE = [
    "scrapper.py",
    "test.txt",
    "test_export.csv",
    "test_model.py",
    "test_model2.py",
    "data_prep.py",
    "prepare_data.py",
    "euromillions.csv",
    "tirages_loto.csv",
    "interface_predictions.html",
    "results_ui.html",
    "gain_calculator.py",
    "model_lstm.py",
    "predict_lstm.py",
    "train_model.py",
    "retrain.py"
]

# Dossiers
LEGACY_DIR = "legacy"
LOG_FILE = "cleanup.log"

# Crée le dossier legacy s’il n’existe pas
os.makedirs(LEGACY_DIR, exist_ok=True)

# Init log
with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"\n--- Archivage lancé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    
    for file in FILES_TO_ARCHIVE:
        if os.path.exists(file):
            target = os.path.join(LEGACY_DIR, file)
            try:
                shutil.move(file, target)
                log.write(f"✅ Archivé : {file} → {target}\n")
                print(f"✅ {file} déplacé vers {LEGACY_DIR}")
            except Exception as e:
                log.write(f"❌ Échec : {file} → {e}\n")
                print(f"❌ Erreur lors du déplacement de {file} : {e}")
        else:
            log.write(f"⚠️ Introuvable : {file}\n")
            print(f"⚠️ {file} non trouvé")

print("\n✅ Nettoyage terminé. Détails dans cleanup.log.")
