# main.py
from data.collector import DataCollector


def main():
    print("🚀 Запуск Finance Bot...")

    # Тестируем сбор данных
    collector = DataCollector()

    # Получаем тестовые данные
    print("\n=== ТЕСТ АКЦИЙ ===")
    stock_data = collector.get_stock_data('AAPL', period='1mo')

    print("\n=== ТЕСТ ВАЛЮТ ===")
    currency_data = collector.get_currency_data(days=10)

    print("\n✅ Все системы работают!")
    print("📊 Следующий шаг: создание моделей ML")


if __name__ == "__main__":
    main()