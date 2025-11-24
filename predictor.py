# predictor.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class DataCollector:
    def __init__(self):
        self.cache = {}

    def get_stock_data(self, symbol, period='1mo'):
        """Получаем данные акций через yfinance"""
        try:
            print(f"📊 Загружаем данные для {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            print(f"✅ Данные {symbol} загружены: {len(data)} записей")
            return data
        except Exception as e:
            print(f"❌ Ошибка загрузки {symbol}: {e}")
            return None

    def get_currency_data(self, base_currency='USD', target_currency='RUB', days=30):
        """Получаем исторические данные курса валют"""
        print("💰 Генерируем данные курса валют...")

        dates = []
        rates = []

        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            # Синтетические данные с трендом + шум
            base_rate = 70 + 5 * (i / days) + 10 * np.sin(i / 30)
            rate = base_rate + np.random.normal(0, 1.5)
            dates.append(date)
            rates.append(round(rate, 2))

        df = pd.DataFrame({
            'date': dates,
            'rate': rates
        })
        print(f"✅ Сгенерировано {len(df)} записей курса валют")
        return df


class CurrencyPredictor:
    def __init__(self):
        self.model = None
        self.mae = None

    def create_features(self, df):
        """Создаем фичи для временного ряда"""
        df = df.copy()

        # Временные фичи
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Лаговые фичи (значения из прошлого)
        df['lag_1'] = df['rate'].shift(1)
        df['lag_2'] = df['rate'].shift(2)
        df['lag_3'] = df['rate'].shift(3)
        df['lag_7'] = df['rate'].shift(7)

        # Скользящие статистики
        df['rolling_mean_3'] = df['rate'].rolling(3).mean()
        df['rolling_std_3'] = df['rate'].rolling(3).std()
        df['rolling_mean_7'] = df['rate'].rolling(7).mean()
        df['rolling_std_7'] = df['rate'].rolling(7).std()

        return df.dropna()

    def train_model(self, currency_data):
        """Обучаем модель на исторических данных"""
        print("🤖 Обучаем модель ML...")

        # Создаем фичи
        df_with_features = self.create_features(currency_data)

        # Подготавливаем данные для обучения
        X = df_with_features.drop(['date', 'rate'], axis=1)
        y = df_with_features['rate']

        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Создаем и обучаем модель
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Оцениваем модель
        predictions = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"✅ Модель обучена!")
        print(f"📊 Точность модели:")
        print(f"   - MAE (Средняя абсолютная ошибка): {self.mae:.2f}")
        print(f"   - RMSE: {rmse:.2f}")

        return self.mae

    def predict_future(self, historical_data, days=7):
        """Прогнозируем на future дней"""
        if self.model is None:
            print("❌ Модель не обучена!")
            return None

        print(f"🔮 Строим прогноз на {days} дней...")

        # Копируем исторические данные для прогноза
        future_predictions = []
        last_data = historical_data.copy()

        for day in range(days):
            # Создаем фичи для последней известной точки
            current_features = self.create_features(last_data).iloc[-1:]
            feature_vector = current_features.drop(['date', 'rate'], axis=1)

            # Предсказываем следующее значение
            next_rate = self.model.predict(feature_vector)[0]
            future_predictions.append(next_rate)

            # Обновляем данные для следующего прогноза
            next_date = last_data['date'].iloc[-1] + timedelta(days=1)
            new_row = pd.DataFrame({
                'date': [next_date],
                'rate': [next_rate]
            })
            last_data = pd.concat([last_data, new_row], ignore_index=True)

        # Создаем даты для прогноза
        future_dates = [historical_data['date'].iloc[-1] + timedelta(days=i + 1) for i in range(days)]

        return future_dates, future_predictions

    def plot_predictions(self, historical_data, future_dates, predictions):
        """Визуализируем историю и прогноз"""
        plt.figure(figsize=(12, 6))

        # Исторические данные
        plt.plot(historical_data['date'], historical_data['rate'],
                 label='Исторические данные', linewidth=2, color='blue')

        # Прогноз
        plt.plot(future_dates, predictions, 'ro--',
                 label='Прогноз', linewidth=2, markersize=6)

        plt.title(f'Прогноз курса USD/RUB на {len(predictions)} дней\nТочность модели (MAE): {self.mae:.2f}')
        plt.xlabel('Дата')
        plt.ylabel('Курс (RUB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохраняем график
        plt.savefig('currency_forecast.png', dpi=100, bbox_inches='tight')
        print("📈 График сохранен как 'currency_forecast.png'")
        plt.show()


# Тестируем модель
if __name__ == "__main__":
    print("🧪 Тестируем модель прогнозирования...")

    # Получаем данные
    collector = DataCollector()
    currency_data = collector.get_currency_data(days=100)  # Больше данных для обучения

    # Обучаем модель
    predictor = CurrencyPredictor()
    mae = predictor.train_model(currency_data)

    # Строим прогноз
    future_dates, predictions = predictor.predict_future(currency_data, days=7)

    # Выводим прогноз
    print("\n📊 ПРОГНОЗ КУРСА USD/RUB:")
    for i, (date, pred) in enumerate(zip(future_dates, predictions), 1):
        print(f"   День {i} ({date.strftime('%d.%m.%Y')}): {pred:.2f} руб.")

    # Строим график
    predictor.plot_predictions(currency_data, future_dates, predictions)