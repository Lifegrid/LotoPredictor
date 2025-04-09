import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def scrap_results():
    url = 'https://www.fdj.fr/jeux-de-tirage/euromillions/resultats'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extraction des boules et des étoiles
    recent_results = soup.find_all('div', class_='result__number')
    boules = [int(num.text) for num in recent_results[:5]]
    etoiles = [int(num.text) for num in recent_results[5:]]
    
    return boules, etoiles

def get_next_draw_date():
    """Retourne la date du prochain tirage Euromillions"""
    today = datetime.now()
    
    # Euromillions se tire tous les mardi et vendredi
    if today.weekday() == 0:  # Lundi
        next_draw = today + timedelta(days=1)  # Prochain tirage mardi
    elif today.weekday() == 3:  # Jeudi
        next_draw = today + timedelta(days=3)  # Prochain tirage vendredi
    elif today.weekday() == 1:  # Mardi
        next_draw = today + timedelta(days=5)  # Prochain tirage vendredi
    elif today.weekday() == 4:  # Vendredi
        next_draw = today + timedelta(days=3)  # Prochain tirage mardi
    else:
        # Si ce n'est ni un lundi, mardi, jeudi ni vendredi, on prend le prochain vendredi
        next_draw = today + timedelta(days=(4 - today.weekday()) % 7)
    
    return next_draw
