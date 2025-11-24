# bot.py
import io
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import matplotlib.pyplot as plt
from predictor import CurrencyPredictor, DataCollector
from stock_analyzer import StockAnalyzer
import pandas as pd
from datetime import datetime


class FinanceBot:
    def __init__(self, token):
        self.token = token
        self.predictor = CurrencyPredictor()
        self.collector = DataCollector()
        self.stock_analyzer = StockAnalyzer()
        self.currency_data = None

        # Загружаем данные при инициализации
        self.load_data()

    def load_data(self):
        """Загружаем данные для обучения модели"""
        print("📊 Загружаем данные для бота...")
        self.currency_data = self.collector.get_currency_data(days=100)

        # Обучаем модель
        self.predictor.train_model(self.currency_data)
        print("✅ Бот инициализирован и готов к работе!")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 *Finance Predictor Bot*

*Доступные команды:*
/start - Начало работы
/forecast - Прогноз курса USD/RUB на 7 дней
/stocks - Информация об акциях
/analyze - Анализ акций
/help - Помощь

*Пример:* Отправь /forecast чтобы получить прогноз курса доллара!
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📋 *Помощь по командам:*

/forecast - Прогноз курса USD/RUB на 7 дней
/stocks - Текущие цены популярных акций
/analyze SYMBOL - Технический анализ акции (пример: /analyze AAPL)
/help - Показать это сообщение

*Примеры:*
/analyze AAPL - анализ Apple
/analyze TSLA - анализ Tesla
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def forecast(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Прогноз курса валют"""
        await update.message.reply_text("🔮 Строю прогноз курса USD/RUB...")

        try:
            # Получаем прогноз
            future_dates, predictions = self.predictor.predict_future(
                self.currency_data, days=7
            )

            # Создаем текст прогноза
            forecast_text = "📊 Прогноз USD/RUB на 7 дней:\n\n"
            for i, (date, pred) in enumerate(zip(future_dates, predictions), 1):
                forecast_text += f"• День {i} ({date.strftime('%d.%m')}): {pred:.2f} руб.\n"

            forecast_text += f"\n📈 Точность модели: MAE = {self.predictor.mae:.2f} пунктов"
            forecast_text += "\n\n⚠️ Прогноз仅供参考"

            # Создаем график
            plot_buffer = self.create_forecast_plot(future_dates, predictions)

            # Отправляем сообщение с графиком
            await update.message.reply_photo(
                photo=InputFile(plot_buffer, filename='forecast.png'),
                caption=forecast_text
            )

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def stocks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Информация об акциях"""
        await update.message.reply_text("📈 Загружаю данные по акциям...")

        try:
            # Получаем данные по популярным акциям
            symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
            stocks_text = "📊 Текущие цены акций:\n\n"

            for symbol in symbols:
                stock_data = self.collector.get_stock_data(symbol, period='1d')
                if stock_data is not None and not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    change = stock_data['Close'].iloc[-1] - stock_data['Open'].iloc[-1]
                    change_percent = (change / stock_data['Open'].iloc[-1]) * 100

                    trend = "📈" if change >= 0 else "📉"
                    stocks_text += f"{trend} {symbol}: ${current_price:.2f} "
                    stocks_text += f"({change:+.2f}, {change_percent:+.1f}%)\n"

            stocks_text += "\nДанные обновлены сегодня"
            await update.message.reply_text(stocks_text)

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка загрузки акций: {e}")

    async def analyze_stock(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Анализ акции с техническими индикаторами"""
        if not context.args:
            await update.message.reply_text(
                "📊 Использование: /analyze SYMBOL\n\n"
                "Пример: /analyze AAPL\n"
                "Доступные символы: AAPL, TSLA, GOOGL, MSFT, AMZN, META"
            )
            return

        symbol = context.args[0].upper()
        await update.message.reply_text(f"📈 Анализирую {symbol}...")

        try:
            # Получаем данные акции
            stock_data = self.collector.get_stock_data(symbol, period='3mo')
            if stock_data is None or stock_data.empty:
                await update.message.reply_text(f"❌ Не удалось загрузить данные для {symbol}")
                return

            # Генерируем сигналы
            signals = self.stock_analyzer.generate_signals(stock_data)

            # Создаем анализ
            analysis_text = f"📊 Технический анализ {symbol}:\n\n"
            for signal in signals:
                analysis_text += f"• {signal}\n"

            # Получаем текущую цену
            current_price = stock_data['Close'].iloc[-1]
            prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100

            analysis_text += f"\n💵 Текущая цена: ${current_price:.2f}\n"
            analysis_text += f"📈 Изменение: {change:+.2f} ({change_percent:+.1f}%)\n"

            # Создаем график
            plot_buffer = self.stock_analyzer.create_stock_analysis_plot(symbol, stock_data)

            await update.message.reply_photo(
                photo=InputFile(plot_buffer, filename=f'analysis_{symbol}.png'),
                caption=analysis_text
            )

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка анализа: {e}")

    async def set_bot_commands(self, application):
        """Устанавливает меню команд в боте"""
        commands = [
            ("start", "Начало работы"),
            ("forecast", "Прогноз курса USD/RUB"),
            ("stocks", "Цены акций"),
            ("analyze", "Анализ акции"),
            ("help", "Помощь по командам")
        ]

        await application.bot.set_my_commands(commands)

    async def handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает неправильные сообщения"""
        help_text = """
🤔 Я не понимаю эту команду.

📋 Доступные команды:
/forecast - Прогноз курса USD/RUB
/stocks - Текущие цены акций  
/analyze SYMBOL - Анализ акции
/help - Помощь по командам

Примеры:
/analyze AAPL - анализ Apple
/forecast - прогноз курса доллара

Или нажми на меню команд слева от поля ввода ↘️
        """
        await update.message.reply_text(help_text)

    def create_forecast_plot(self, future_dates, predictions):
        """Создает график прогноза для отправки в Telegram"""
        plt.figure(figsize=(10, 6))

        # Последние 30 дней исторических данных
        historical = self.currency_data.tail(30)

        # Исторические данные
        plt.plot(historical['date'], historical['rate'],
                 label='Исторические данные', linewidth=2, color='blue')

        # Прогноз
        plt.plot(future_dates, predictions, 'ro--',
                 label='Прогноз', linewidth=2, markersize=6)

        plt.title('Прогноз курса USD/RUB на 7 дней')
        plt.xlabel('Дата')
        plt.ylabel('Курс (RUB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохраняем в buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def run(self):
        """Запускает бота"""
        application = Application.builder().token(self.token).build()

        # Регистрируем обработчики команд
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("forecast", self.forecast))
        application.add_handler(CommandHandler("stocks", self.stocks))
        application.add_handler(CommandHandler("analyze", self.analyze_stock))

        # Обработчик для неправильных сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_unknown))

        # Устанавливаем меню команд
        application.post_init = self.set_bot_commands

        print("🤖 Бот запущен! Нажми Ctrl+C для остановки.")
        application.run_polling()


# Основная функция
def main():
    # Твой токен бота
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Сначала получи токен бота от @BotFather и вставь его в код!")
        return

    # Создаем и запускаем бота
    bot = FinanceBot(BOT_TOKEN)
    bot.run()


if __name__ == "__main__":
    main()