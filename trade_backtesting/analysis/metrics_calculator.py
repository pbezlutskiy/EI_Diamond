# analysis/metrics_calculator.py
"""Расчет метрик бэктеста"""
import numpy as np


class MetricsCalculator:
    """Расчет всех метрик стратегии"""
    
    @staticmethod
    def calculate_all_metrics(trades: list, equity_curve: list) -> dict:
        """Рассчитывает все метрики"""
        profits = [t['profit'] for t in trades]
        
        total_profit = sum(profits)
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        win_rate = len(wins) / len(profits) if profits else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        
        # Drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio (упрощенный)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'total_profit': total_profit,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
