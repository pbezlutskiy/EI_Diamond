# analysis/probability_calculator.py
"""Вероятностные метрики"""


class ProbabilityCalculator:
    """Расчет вероятностей успеха"""
    
    @staticmethod
    def calculate_probabilities(trades: list) -> dict:
        """Рассчитывает вероятности"""
        profits = [t['profit'] for t in trades]
        
        p_profit = len([p for p in profits if p > 0]) / len(profits) if profits else 0
        p_loss = 1 - p_profit
        
        # Вероятность последовательных убытков
        max_consecutive_losses = 0
        current_streak = 0
        for p in profits:
            if p < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        return {
            'p_profit': p_profit,
            'p_loss': p_loss,
            'max_consecutive_losses': max_consecutive_losses
        }
