# result_viewer/result_to_html.py
"""Экспорт результатов в HTML"""
from result_viewer.base_viewer import IResultViewer


class ResultToHTML(IResultViewer):
    """Сохраняет результаты в HTML"""
    
    def __init__(self):
        self.results = []
    
    def view(self, test_result) -> None:
        """Добавляет результат"""
        self.results.append(test_result)
    
    def save_to_file(self, filename: str):
        """Сохраняет все результаты в HTML"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Backtest Results</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>📊 Backtest Results</h1>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Total Profit</th>
            <th>Win Rate</th>
            <th>Max DD</th>
            <th>Sharpe</th>
        </tr>"""
        
        for result in self.results:
            html += f"""
        <tr>
            <td>{result.get('strategy_name', 'Unknown')}</td>
            <td>{result.get('total_profit', 0):.2f}</td>
            <td>{result.get('win_rate', 0)*100:.1f}%</td>
            <td>{result.get('max_drawdown', 0)*100:.1f}%</td>
            <td>{result.get('sharpe_ratio', 0):.2f}</td>
        </tr>"""
        
        html += """
    </table>
</body>
</html>"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
