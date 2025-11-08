# ml_analysis/ml_validator.py
"""Валидация ML моделей"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


class MLValidator:
    """Валидирует ML модель"""
    
    @staticmethod
    def cross_validate(model, X, y, cv: int = 5) -> dict:
        """Кросс-валидация"""
        scores = cross_val_score(model, X, y, cv=cv)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    @staticmethod
    def evaluate(model, X_test, y_test) -> str:
        """Оценка на тестовой выборке"""
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)
