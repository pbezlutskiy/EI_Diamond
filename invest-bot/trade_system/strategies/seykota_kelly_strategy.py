# Путь: invest-bot/trade_system/strategies/seykota_kelly_strategy.py
"""
Стратегия Эда Сейкоты с формулой Келли
ИСПРАВЛЕНА: логика стоп-лосса + минимальная дистанция до стопа + корректная типизация
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from trade_system.signal import Signal, SignalType

logger = logging.getLogger(__name__)


class SeykoaKellyStrategy:
    """Трендовая стратегия с управлением капиталом по Келли"""
    
    def __init__(self, figi: str, ticker: str, settings: Dict[str, Any]):
        """
        Инициализация стратегии
        
        Args:
            figi: идентификатор инструмента (Тинькофф)
            ticker: тикер инструмента (например, VTBR)
            settings: словарь с параметрами стратегии
        """
        self.figi = figi
        self.ticker = ticker
        
        # Параметры EMA (экспоненциальная скользящая средняя)
        self.ema_short_period = int(settings.get("EMA_SHORT", "18"))
        self.ema_long_period = int(settings.get("EMA_LONG", "50"))
        self.min_candles = int(settings.get("MIN_CANDLES", "100"))
        
        # Параметры стоп-лосса
        # ATR (Average True Range) - показатель волатильности
        self.atr_period = int(settings.get("ATR_PERIOD", "14"))
        # Множитель для ATR - дальность стопа от ATR
        self.atr_multiplier = float(settings.get("ATR_MULTIPLIER", "2.0"))
        # Минимальная дистанция стопа в единицах ATR (от цены входа)
        # По умолчанию 3.0 * ATR - это означает, что стоп будет минимум на 3 ATR от цены входа
        self.min_atr_distance = float(settings.get("MIN_ATR_DISTANCE", "3.0"))
        # Процентный стоп (7% от цены входа)
        self.percent_stop = float(settings.get("PERCENT_STOP", "0.07"))
        # Swing period - количество свечей для поиска локального экстремума (min/max)
        self.swing_period = int(settings.get("SWING_PERIOD", "40"))
        
        # Параметры Келли (управление капиталом)
        self.kelly_fraction = float(settings.get("KELLY_FRACTION", "0.25"))
        self.max_risk = float(settings.get("MAX_RISK", "0.05"))
        self.min_trades_for_kelly = int(settings.get("MIN_TRADES_FOR_KELLY", "30"))
        
        # История сделок для расчета Келли
        self.trade_history = []
        
        logger.info(f"SeykoaKellyStrategy initialized for {ticker}")
        logger.info(f"  EMA periods: {self.ema_short_period}/{self.ema_long_period}")
        logger.info(f"  ATR: period={self.atr_period}, multiplier={self.atr_multiplier}, min_distance={self.min_atr_distance}")
        logger.info(f"  Swing period: {self.swing_period}")
    
    @staticmethod
    def _to_float(value) -> float:
        """
        Конвертирует Quotation из Tinkoff API или любое значение в float
        
        Args:
            value: значение (может быть Quotation с units и nano, или обычное число)
            
        Returns:
            float: преобразованное значение
        """
        if hasattr(value, 'units') and hasattr(value, 'nano'):
            # Это Quotation объект из Tinkoff API
            return float(value.units) + float(value.nano) / 1e9
        return float(value)
    
    def analyze_candles(self, candles: list) -> Optional[Signal]:
        """
        Анализирует последовательность свечей и возвращает торговый сигнал
        
        Args:
            candles: список HistoricCandle объектов
            
        Returns:
            Signal или None: сигнал на открытие позиции (LONG/SHORT) или None
        """
        if len(candles) < self.min_candles:
            return None
        
        # Расчет индикаторов (с конвертацией Quotation в float)
        closes = [self._to_float(c.close) for c in candles]
        ema_short = self._calculate_ema(closes, self.ema_short_period)
        ema_long = self._calculate_ema(closes, self.ema_long_period)
        atr = self._calculate_atr(candles, self.atr_period)
        
        # Текущие значения индикаторов
        current_ema_short = ema_short[-1]
        current_ema_long = ema_long[-1]
        prev_ema_short = ema_short[-2]
        prev_ema_long = ema_long[-2]
        current_price = closes[-1]
        current_atr = atr[-1]
        
        # Определение тренда через пересечение EMA
        signal_type = None
        
        # LONG: короткая EMA пересекает длинную снизу вверх (золотой крест)
        if prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long:
            signal_type = SignalType.LONG
        
        # SHORT: короткая EMA пересекает длинную сверху вниз (смертельный крест)
        elif prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long:
            signal_type = SignalType.SHORT
        
        if signal_type is None:
            return None
        
        # Расчет уровня стоп-лосса на основе swing-экстремума и ATR
        swing_level = self._calculate_swing_level(candles, signal_type)
        stop_loss = self._calculate_stop_loss(
            signal_type, 
            current_price, 
            current_atr, 
            swing_level
        )
        
        return Signal(
            signal_type=signal_type,
            figi=self.figi,
            ticker=self.ticker,
            price=Decimal(str(current_price)),
            stop_loss_level=Decimal(str(stop_loss)),
            timestamp=datetime.now()
        )
    
    def _calculate_ema(self, prices: list, period: int) -> list:
        """
        Рассчитывает экспоненциальную скользящую среднюю (EMA)
        
        Args:
            prices: список цен закрытия
            period: период EMA
            
        Returns:
            list: значения EMA
        """
        ema = [prices[0]]
        k = 2 / (period + 1)  # коэффициент сглаживания
        for price in prices[1:]:
            ema_value = price * k + ema[-1] * (1 - k)
            ema.append(ema_value)
        return ema
    
    def _calculate_atr(self, candles: list, period: int) -> list:
        """
        Рассчитывает среднее истинное размах (ATR - Average True Range)
        ATR показывает волатильность рынка
        
        Args:
            candles: список HistoricCandle объектов
            period: период ATR
            
        Returns:
            list: значения ATR
        """
        atr_values = [0]
        
        for i in range(1, len(candles)):
            high = self._to_float(candles[i].high)
            low = self._to_float(candles[i].low)
            prev_close = self._to_float(candles[i-1].close)
            
            # Истинный размах (True Range)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            
            # EMA формула для ATR
            if i < period:
                # На начальном этапе - простое среднее
                atr_values.append(sum([atr_values[j] if j == 0 else tr for j in range(i)]) / i)
            else:
                # После периода - EMA
                prev_atr = atr_values[-1]
                atr_values.append((prev_atr * (period - 1) + tr) / period)
        
        return atr_values
    
    def _calculate_swing_level(self, candles: list, signal_type: SignalType) -> float:
        """
        Находит локальный экстремум (swing) за период
        
        Args:
            candles: список свечей
            signal_type: тип сигнала (LONG или SHORT)
            
        Returns:
            float: уровень swing (для LONG - минимум, для SHORT - максимум)
        """
        recent_candles = candles[-self.swing_period:]
        
        if signal_type == SignalType.LONG:
            # Для LONG: ищем swing low (минимальная цена за период)
            return min(self._to_float(c.low) for c in recent_candles)
        else:
            # Для SHORT: ищем swing high (максимальная цена за период)
            return max(self._to_float(c.high) for c in recent_candles)
    
    def _calculate_stop_loss(self, signal_type: SignalType, entry_price: float, 
                            atr: float, swing_level: float) -> float:
        """
        Рассчитывает уровень стоп-лосса с защитой от слишком близких уровней
        
        Используется гибридный подход:
        1. ATR-based stop: ATR * multiplier от цены входа
        2. Percent-based stop: процент от цены входа
        3. Swing-based stop: локальный экстремум за период
        4. Min distance protection: минимум MIN_ATR_DISTANCE * ATR от входа
        
        Args:
            signal_type: тип позиции (LONG или SHORT)
            entry_price: цена входа
            atr: текущее значение ATR
            swing_level: уровень swing-экстремума
            
        Returns:
            float: оптимальный уровень стоп-лосса
        """
        
        if signal_type == SignalType.LONG:
            # ===== ДЛЯ LONG ПОЗИЦИИ =====
            # Стоп должен быть НИЖЕ цены входа
            
            # 1. ATR-based: цена - (ATR * множитель)
            atr_stop = entry_price - (atr * self.atr_multiplier)
            
            # 2. Percent-based: цена * (1 - процент)
            percent_stop = entry_price * (1 - self.percent_stop)
            
            # 3. Swing-based: минимум за период
            swing_stop = swing_level
            
            # 4. Минимальная дистанция: цена - (ATR * MIN_ATR_DISTANCE)
            # Это гарантирует, что стоп не будет слишком близко к входу
            min_distance_stop = entry_price - (atr * self.min_atr_distance)
            
            # Берем МАКСИМУМ (ближайший к цене, но не слишком рядом)
            hybrid_stop = max(atr_stop, percent_stop, swing_stop, min_distance_stop)
            
            logger.info(
                f"LONG STOP: ATR={atr_stop:.2f}, Percent={percent_stop:.2f}, "
                f"Swing={swing_stop:.2f}, MinDist={min_distance_stop:.2f}, HYBRID={hybrid_stop:.2f}"
            )
            return hybrid_stop
        
        else:  # SHORT
            # ===== ДЛЯ SHORT ПОЗИЦИИ =====
            # Стоп должен быть ВЫШЕ цены входа
            
            # 1. ATR-based: цена + (ATR * множитель)
            atr_stop = entry_price + (atr * self.atr_multiplier)
            
            # 2. Percent-based: цена * (1 + процент)
            percent_stop = entry_price * (1 + self.percent_stop)
            
            # 3. Swing-based: максимум за период
            swing_stop = swing_level
            
            # 4. Минимальная дистанция: цена + (ATR * MIN_ATR_DISTANCE)
            # Это гарантирует, что стоп не будет слишком близко к входу
            min_distance_stop = entry_price + (atr * self.min_atr_distance)
            
            # Берем МИНИМУМ (ближайший к цене, но не слишком рядом)
            hybrid_stop = min(atr_stop, percent_stop, swing_stop, min_distance_stop)
            
            logger.info(
                f"SHORT STOP: ATR={atr_stop:.2f}, Percent={percent_stop:.2f}, "
                f"Swing={swing_stop:.2f}, MinDist={min_distance_stop:.2f}, HYBRID={hybrid_stop:.2f}"
            )
            return hybrid_stop
    
    def record_trade_result(self, signal_type: SignalType, entry_price: float, 
                           exit_price: float, entry_time: datetime, exit_time: datetime):
        """
        Записывает результат закрытой сделки для расчета статистики Келли
        
        Args:
            signal_type: тип позиции
            entry_price: цена входа
            exit_price: цена выхода
            entry_time: время входа
            exit_time: время выхода
        """
        
        if signal_type == SignalType.LONG:
            profit = exit_price - entry_price
        else:
            profit = entry_price - exit_price
        
        self.trade_history.append({
            'signal_type': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'win': profit > 0
        })
    
    def _kelly_calculator(self) -> float:
        """
        Рассчитывает оптимальную долю капитала по формуле Келли
        Формула Келли: f* = (bp - q) / b
        где: b - отношение выигрыша, p - вероятность выигрыша, q - вероятность проигрыша
        
        Returns:
            float: оптимальная доля капитала для ставки
        """
        if len(self.trade_history) < self.min_trades_for_kelly:
            return self.kelly_fraction
        
        wins = [t for t in self.trade_history if t['win']]
        losses = [t for t in self.trade_history if not t['win']]
        
        if not wins or not losses:
            return self.kelly_fraction
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = sum(t['profit'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
        
        if avg_win == 0:
            return self.kelly_fraction
        
        # Формула Келли
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Ограничение максимальным риском
        kelly_pct = min(max(kelly_pct, 0), self.max_risk)
        
        # Консервативная доля (fraction - для уменьшения агрессивности)
        return kelly_pct * self.kelly_fraction
