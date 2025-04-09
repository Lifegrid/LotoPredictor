import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Définir la fonction 'mse' avant de charger le modèle
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Charger le modèle
model = load_model('ai/models/hybrid_model.h5', custom_objects={'mse': mse})

# Affichage de l'architecture du modèle
model.summary()

# Exemple de données réelles pour tester la prédiction
last_results = [
    {"boules": [1, 12, 23, 35, 47], "etoiles": [4, 9]},
    {"boules": [2, 14, 22, 36, 48], "etoiles": [3, 10]},
    {"boules": [3, 5, 8, 19, 42], "etoiles": [2, 6]},
    {"boules": [4, 11, 17, 30, 44], "etoiles": [1, 7]},
    {"boules": [5, 9, 13, 24, 46], "etoiles": [8, 9]},
    {"boules": [6, 10, 15, 28, 49], "etoiles": [4, 11]},
    {"boules": [7, 12, 20, 34, 50], "etoiles": [5, 12]},
    {"boules": [8, 16, 21, 38, 47], "etoiles": [1, 2]},
    {"boules": [9, 18, 25, 40, 44], "etoiles": [3, 8]},
    {"boules": [10, 13, 22, 36, 48], "etoiles": [7, 9]}
]

# Préparer les entrées pour le modèle (boules et étoiles)
boules_input = np.array([r['boules'] for r in last_results])
etoiles_input = np.array([r['etoiles'] for r in last_results])

# Ajouter la dimension du batch en premier (forme : (1, 10, 5) pour boules, (1, 10, 2) pour étoiles)
boules_input = boules_input[np.newaxis, ...]  # Ajouter la dimension du batch pour boules
etoiles_input = etoiles_input[np.newaxis, ...]  # Ajouter la dimension du batch pour étoiles

# Effectuer une prédiction
predictions = model.predict([boules_input, etoiles_input])

# Affichage des prédictions
print("Prédictions :", predictions)

# Afficher les résultats sous forme de boules et étoiles
# Exemple de post-traitement si nécessaire
numbers = np.clip(np.round(predictions[0]), 1, 50).astype(int)
stars = np.clip(np.round(predictions[1]), 1, 12).astype(int)

print("Prédiction des boules :", numbers)
print("Prédiction des étoiles :", stars)
