# detailed_backtest_visualizer.py
"""Детальная визуализация бэктеста с EMA, ATR и точными входами/выходами"""
import pandas as pd
from pathlib import Path
from datetime import datetime


class DetailedBacktestVisualizer:
    """Создает подробный HTML отчет с графиками сделок, индикаторами EMA и ATR"""
    
    @staticmethod
    def calculate_ema(prices, period):
        """Рассчитывает EMA"""
        ema = [prices[0]]
        k = 2 / (period + 1)
        for price in prices[1:]:
            ema.append(price * k + ema[-1] * (1 - k))
        return ema
    
    @staticmethod
    def calculate_atr(candles_data, period=14):
        """Рассчитывает ATR"""
        atr_values = [0]
        for i in range(1, len(candles_data)):
            high = candles_data[i]['high']
            low = candles_data[i]['low']
            prev_close = candles_data[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            
            if i < period:
                atr_values.append(sum([atr_values[j] if j == 0 else tr for j in range(i)]) / i)
            else:
                prev_atr = atr_values[-1]
                atr_values.append((prev_atr * (period - 1) + tr) / period)
        
        return atr_values
    
    @staticmethod
    def normalize_time(time_value):
        """Нормализует время к datetime"""
        if isinstance(time_value, str):
            return pd.to_datetime(time_value)
        elif isinstance(time_value, (datetime, pd.Timestamp)):
            return pd.to_datetime(time_value)
        else:
            return time_value
    
    @staticmethod
    def generate_detailed_report(ticker: str, candles_data: list, trades: list, 
                                equity: list, metrics: dict, output_file: str):
        """Генерирует детальный HTML отчет с EMA, ATR и точными входами"""
        
        print(f"\n🔍 DEBUG Визуализация:")
        print(f"   Свечей: {len(candles_data)}")
        print(f"   Сделок: {len(trades)}")
        
        # Подготовка данных
        df = pd.DataFrame(candles_data)
        df['index'] = range(len(df))
        
        # Нормализация времени
        df['time_norm'] = df['time'].apply(DetailedBacktestVisualizer.normalize_time)
        
        print(f"   Первая свеча: {df['time_norm'].iloc[0]}")
        print(f"   Последняя свеча: {df['time_norm'].iloc[-1]}")
        
        # Расчет индикаторов
        ema18_values = DetailedBacktestVisualizer.calculate_ema(df['close'].tolist(), 18)
        ema50_values = DetailedBacktestVisualizer.calculate_ema(df['close'].tolist(), 50)
        atr_values = DetailedBacktestVisualizer.calculate_atr(candles_data, 14)
        
        # ТОЧНЫЙ поиск входов/выходов по времени
        entry_points = []
        exit_points = []
        
        for i, trade in enumerate(trades, 1):
            if 'entry_time' in trade:
                trade_entry_norm = DetailedBacktestVisualizer.normalize_time(trade['entry_time'])
                
                # Поиск ТОЧНОГО совпадения
                entry_match = df[df['time_norm'] == trade_entry_norm]
                
                if not entry_match.empty:
                    entry_idx = entry_match.index[0]
                else:
                    # Ближайшее время
                    df['time_diff'] = abs(df['time_norm'] - trade_entry_norm)
                    entry_idx = df['time_diff'].idxmin()
                    print(f"⚠️ Сделка #{i}: Точное совпадение не найдено, использую ближайшее")
                
                # ПРОВЕРКА совпадения цены
                actual_price = df.loc[entry_idx, 'close']
                if abs(actual_price - trade['entry_price']) > 5:
                    print(f"⚠️ НЕСООТВЕТСТВИЕ Сделка #{i}: entry_price={trade['entry_price']:.2f}, график={actual_price:.2f}, diff={abs(actual_price - trade['entry_price']):.2f}")
            else:
                entry_idx = i * 100
                print(f"⚠️ Сделка #{i}: НЕТ entry_time!")
            
            # Поиск выхода
            if 'exit_time' in trade:
                trade_exit_norm = DetailedBacktestVisualizer.normalize_time(trade['exit_time'])
                exit_match = df[df['time_norm'] == trade_exit_norm]
                
                if not exit_match.empty:
                    exit_idx = exit_match.index[0]
                else:
                    df['time_diff'] = abs(df['time_norm'] - trade_exit_norm)
                    exit_idx = df['time_diff'].idxmin()
            else:
                exit_idx = entry_idx + 50
            
            entry_points.append({
                'x': int(entry_idx),
                'y': float(trade['entry_price']),
                'trade_num': i,
                'type': str(trade.get('signal_type', 'LONG'))
            })
            
            exit_points.append({
                'x': int(exit_idx),
                'y': float(trade['exit_price']),
                'trade_num': i,
                'profit': float(trade['profit']),
                'reason': trade.get('reason', 'unknown')
            })
        
        print(f"✅ Найдено точек входа: {len(entry_points)}")
        print(f"✅ Найдено точек выхода: {len(exit_points)}")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{ticker} - Детальный Анализ</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
        }}
        h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 40px;
        }}
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .chart {{ margin: 30px 0; background: #f8f9fa; padding: 20px; border-radius: 10px; }}
        .section-title {{ font-size: 24px; margin: 30px 0 15px 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 {ticker} - Детальный Анализ</h1>
    
    <div class="summary">
        <div class="metric"><strong>Total Profit</strong><div style="font-size:28px; font-weight:bold; color:{'#22c55e' if metrics['total_profit'] > 0 else '#ef4444'}">{metrics['total_profit']:.2f} ₽</div></div>
        <div class="metric"><strong>Win Rate</strong><div style="font-size:28px; font-weight:bold">{metrics['win_rate']*100:.1f}%</div></div>
        <div class="metric"><strong>Profit Factor</strong><div style="font-size:28px; font-weight:bold">{metrics['profit_factor']:.2f}</div></div>
        <div class="metric"><strong>Max Drawdown</strong><div style="font-size:28px; font-weight:bold">{metrics['max_drawdown']*100:.1f}%</div></div>
        <div class="metric"><strong>Sharpe</strong><div style="font-size:28px; font-weight:bold">{metrics['sharpe_ratio']:.2f}</div></div>
        <div class="metric"><strong>Total Trades</strong><div style="font-size:28px; font-weight:bold">{metrics['total_trades']}</div></div>
    </div>
    
    <h2 class="section-title">📈 График Цены с EMA(18), EMA(50)</h2>
    <div id="price-chart" class="chart"></div>
    
    <h2 class="section-title">📊 ATR(14)</h2>
    <div id="atr-chart" class="chart"></div>
    
    <h2 class="section-title">💰 Кривая Капитала</h2>
    <div id="equity-chart" class="chart"></div>
</div>

<script>
var priceData = {{
    x: {list(range(len(df)))},
    y: {df['close'].tolist()},
    type: 'scatter',
    mode: 'lines',
    name: 'Цена',
    line: {{ color: '#3b82f6', width: 1.5 }}
}};

var ema18Data = {{
    x: {list(range(len(df)))},
    y: {ema18_values},
    type: 'scatter',
    mode: 'lines',
    name: 'EMA(18)',
    line: {{ color: '#22c55e', width: 2 }}
}};

var ema50Data = {{
    x: {list(range(len(df)))},
    y: {ema50_values},
    type: 'scatter',
    mode: 'lines',
    name: 'EMA(50)',
    line: {{ color: '#ef4444', width: 2 }}
}};

var entryLongs = {{
    x: {[p['x'] for p in entry_points if 'LONG' in p['type']]},
    y: {[p['y'] for p in entry_points if 'LONG' in p['type']]},
    mode: 'markers',
    name: 'LONG',
    marker: {{ size: 12, color: '#22c55e', symbol: 'triangle-up', line: {{ color: 'white', width: 2 }} }},
    text: {[f"#{p['trade_num']}" for p in entry_points if 'LONG' in p['type']]},
    hovertemplate: '<b>LONG</b><br>%{{text}}<br>%{{y:.2f}} ₽<extra></extra>'
}};

var entryShorts = {{
    x: {[p['x'] for p in entry_points if 'SHORT' in p['type']]},
    y: {[p['y'] for p in entry_points if 'SHORT' in p['type']]},
    mode: 'markers',
    name: 'SHORT',
    marker: {{ size: 12, color: '#ef4444', symbol: 'triangle-down', line: {{ color: 'white', width: 2 }} }}
}};

var exitStops = {{
    x: {[p['x'] for p in exit_points if p['reason'] == 'stop']},
    y: {[p['y'] for p in exit_points if p['reason'] == 'stop']},
    mode: 'markers',
    name: 'СТОП',
    marker: {{ size: 10, color: '#f59e0b', symbol: 'x', line: {{ width: 2 }} }}
}};

var exitEnds = {{
    x: {[p['x'] for p in exit_points if p['reason'] == 'end']},
    y: {[p['y'] for p in exit_points if p['reason'] == 'end']},
    mode: 'markers',
    name: 'КОНЕЦ',
    marker: {{ size: 10, color: '#3b82f6', symbol: 'square' }}
}};

Plotly.newPlot('price-chart', [priceData, ema18Data, ema50Data, entryLongs, entryShorts, exitStops, exitEnds], {{
    title: 'Цена, EMA(18), EMA(50) с Входами/Выходами',
    xaxis: {{ title: 'Свеча #' }},
    yaxis: {{ title: 'Цена (₽)' }},
    hovermode: 'closest',
    height: 600
}});

Plotly.newPlot('atr-chart', [{{
    x: {list(range(len(atr_values)))},
    y: {atr_values},
    type: 'scatter',
    mode: 'lines',
    line: {{ color: '#8b5cf6', width: 2 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(139, 92, 246, 0.3)'
}}], {{
    title: 'ATR(14)',
    xaxis: {{ title: 'Свеча #' }},
    yaxis: {{ title: 'ATR' }},
    height: 300
}});

Plotly.newPlot('equity-chart', [{{
    y: {equity},
    type: 'scatter',
    mode: 'lines',
    line: {{ color: '#22c55e', width: 3 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(34, 197, 94, 0.2)'
}}], {{
    title: 'Капитал',
    xaxis: {{ title: 'Сделка #' }},
    yaxis: {{ title: '₽' }},
    height: 400
}});
</script>
</body>
</html>"""
        
        Path("backtest_results").mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Детальный отчет создан: {output_file}")
