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

# Crée des données factices pour tester la prédiction
dummy_boules_input = np.random.rand(1, 10, 5)  # Forme (1, 10, 5) pour les boules
dummy_etoiles_input = np.random.rand(1, 10, 2)  # Forme (1, 10, 2) pour les étoiles

# Effectuer une prédiction
predictions = model.predict([dummy_boules_input, dummy_etoiles_input])

# Affichage des prédictions
print("Prédictions :", predictions)
