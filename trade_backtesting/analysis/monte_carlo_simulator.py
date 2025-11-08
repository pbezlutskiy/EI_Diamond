# analysis/monte_carlo_simulator.py
"""Monte Carlo симуляция"""
import numpy as np
import random


class MonteCarloSimulator:
    """Monte Carlo для оценки робастности"""
    
    @staticmethod
    def run_simulation(trades: list, iterations: int = 1000) -> dict:
        """Запускает Monte Carlo симуляцию"""
        profits = [t['profit'] for t in trades]
        
        results = []
        
        for _ in range(iterations):
            # Перемешиваем порядок сделок
            shuffled = random.sample(profits, len(profits))
            
            # Считаем итоговую прибыль
            total = sum(shuffled)
            results.append(total)
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_95': np.percentile(results, 95)
        }
