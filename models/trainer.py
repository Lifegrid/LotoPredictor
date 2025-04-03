import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from data_prep import load_and_prepare_data

# 📥 Chargement des données
X, y, scaler = load_and_prepare_data("euromillions_clean.csv")

# 🏗️ Construction du modèle
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')

# 🧠 Entraînement
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = model.fit(X, y, epochs=100, batch_size=32, verbose=1, callbacks=[early_stop])

# 💾 Sauvegarde du modèle
model.save("models/euromillions_lstm.h5")

# 📊 Visualisation de la perte
plt.plot(history.history['loss'])
plt.title("Courbe de perte (loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid()
plt.savefig("models/training_loss.png")
plt.show()

print("✅ Modèle entraîné et sauvegardé dans 'models/euromillions_lstm.h5'")
