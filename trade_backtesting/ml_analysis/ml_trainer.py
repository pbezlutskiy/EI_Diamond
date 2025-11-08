# ml_analysis/ml_trainer.py
"""Обучение ML моделей"""
from sklearn.ensemble import RandomForestClassifier
import pickle


class MLTrainer:
    """Обучает ML модель на исторических данных"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
    
    def train(self, X_train: list, y_train: list):
        """Обучает модель"""
        self.model.fit(X_train, y_train)
        return self.model
    
    def save_model(self, filepath: str):
        """Сохраняет модель"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        """Загружает модель"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
