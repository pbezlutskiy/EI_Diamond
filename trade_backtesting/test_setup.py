import sys
sys.path.append('../invest-bot')

try:
    from tinkoff.invest import Client
    print("✅ tinkoff-investments установлен")
except:
    print("❌ Установите: pip install tinkoff-investments")

try:
    import pandas
    print("✅ pandas установлен")
except:
    print("❌ Установите: pip install pandas")

try:
    import numpy
    print("✅ numpy установлен")
except:
    print("❌ Установите: pip install numpy")

try:
    from trade_system.strategies.seykota_kelly_strategy import SeykoaKellyStrategy
    print("✅ Стратегия импортируется")
except Exception as e:
    print(f"❌ Ошибка импорта стратегии: {e}")

print("\n🎉 Готово к запуску!")
