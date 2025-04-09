# 🎯 EuroMillions Predictor Pro

Un projet complet d'intelligence artificielle combinant LSTM, statistiques et optimisation génétique pour générer les meilleures grilles prédictives EuroMillions.  
Optimisé pour un usage local, une interface web intuitive, et une logique de prédiction avancée, fiable et évolutive.

---

## 🚀 Fonctionnalités

- 🔮 Prédictions EuroMillions avec modèle LSTM entraîné sur données réelles
- 🧬 Générateur de grilles par sélection génétique
- 📊 Visualisation des fréquences des numéros / étoiles
- 🔁 Historique des résultats et prédictions
- 🧠 Réentraînement automatique du modèle
- 🌐 Interface web Flask responsive avec style CSS pro
- 📁 Structure modulaire et propre pour scalabilité
- 🔧 Scraper FDJ intégré (avec fallback intelligent)

---

## 🗂️ Structure du projet

```bash
LotoPredictor/
├── ai/
│   ├── models/
│   └── trainer.py
├── games/
│   └── euromillions/
│       ├── data_loader.py
│       ├── predictor.py
│       └── genetic_optimizer.py
├── services/
│   ├── fdj_scraper.py
│   ├── history_manager.py
│   └── result_checker.py
├── static/
│   └── styles.css
├── templates/
│   ├── index.html
│   ├── stats.html
│   └── error.html
├── data/
│   └── euromillions_clean.csv
├── .env.example
├── .gitignore
├── app.py
└── README.md


💻 Installation locale
⚙️ Pré-requis
Python 3.10 ou plus

pip, virtualenv

Un terminal administrateur si Windows + droits d’accès

🧪 Étapes
bash
Copier
Modifier
# 1. Cloner le projet
git clone https://github.com/TON-USER/euromillions-predictor-pro.git
cd euromillions-predictor-pro

# 2. Créer et activer l'environnement
python -m venv lotoenv
lotoenv\Scripts\activate     # sous Windows
# ou
source lotoenv/bin/activate  # sous macOS/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'entraînement du modèle
python ai/trainer.py

# 5. Démarrer l'application
python app.py
🌐 Accès Web
Par défaut :

bash
Copier
Modifier
http://127.0.0.1:5000/
📷 Captures d'écran (optionnel)
Ajoute ici des captures pour illustrer les prédictions et l'interface.

🤝 Contributions
Les PR sont les bienvenues ! Pour contribuer :

Fork le repo

Crée une branche (git checkout -b feature/amélioration)

Commit (git commit -am 'feat: amélioration du modèle')

Push (git push origin feature/amélioration)

Crée une Pull Request

🛡️ Licence
Projet sous licence MIT.

🧠 Auteurs
Projet initié et géré par @TON-USER

🔮 Assistance IA : GPT-4, DeepSeek & outils avancés

