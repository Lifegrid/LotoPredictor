import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 📥 Chargement des données
df = pd.read_csv("euromillions_clean.csv")
boule_cols = ["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"]
etoile_cols = ["etoile_1", "etoile_2"]

# 🔢 Fréquence de chaque numéro
boules = df[boule_cols].values.ravel()
etoiles = df[etoile_cols].values.ravel()

boule_counts = Counter(boules)
etoile_counts = Counter(etoiles)

# 🔝 Top 15 boules
top_boules = dict(sorted(boule_counts.items(), key=lambda x: x[1], reverse=True)[:15])
top_etoiles = dict(sorted(etoile_counts.items(), key=lambda x: x[1], reverse=True)[:10])

print("🔢 Boules les plus fréquentes :")
print(top_boules)
print("\n⭐ Étoiles les plus fréquentes :")
print(top_etoiles)

# 📊 Visualisation
def plot_freq(freq_dict, title, max_val):
    plt.figure(figsize=(10, 4))
    plt.bar(freq_dict.keys(), freq_dict.values(), width=0.6)
    plt.title(title)
    plt.xlabel("Numéro")
    plt.ylabel("Fréquence")
    plt.xticks(range(1, max_val + 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

plot_freq(boule_counts, "📊 Fréquence des boules", 50)
plot_freq(etoile_counts, "⭐ Fréquence des étoiles", 12)
