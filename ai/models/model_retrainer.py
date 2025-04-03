def retrain_model():
    from services.scraper import get_full_history
    from ai.trainer import train_model
    
    data = get_full_history()  # À implémenter
    train_model(data)  # Votre script d'entraînement existant
    
    from ai.predictor import Predictor  # Import the Predictor class or module
    predictor = Predictor()  # Initialize the predictor instance
    predictor.reload_model()  # Méthode à ajouter