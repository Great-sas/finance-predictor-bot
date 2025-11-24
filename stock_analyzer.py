# stock_analyzer.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io


class StockAnalyzer:
    def __init__(self):
        pass

    def calculate_technical_indicators(self, data):
        """Рассчитывает технические индикаторы"""
        df = data.copy()

        # SMA (Simple Moving Average)
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # RSI (Relative Strength Index)
        df['RSI'] = self.calculate_rsi(df['Close'])

        # MACD
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])

        return df

    def calculate_rsi(self, prices, period=14):
        """Рассчитывает RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Рассчитывает MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def generate_signals(self, data):
        """Генерирует торговые сигналы"""
        df = self.calculate_technical_indicators(data)
        signals = []

        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_Signal'].iloc[-1]

        # RSI сигналы
        if current_rsi < 30:
            signals.append("📈 RSI: ПЕРЕПРОДАННОСТЬ (возможен рост)")
        elif current_rsi > 70:
            signals.append("📉 RSI: ПЕРЕКУПЛЕННОСТЬ (возможен спад)")
        else:
            signals.append("⚖️ RSI: НЕЙТРАЛЬНЫЙ")

        # MACD сигналы
        if current_macd > current_signal:
            signals.append("✅ MACD: БЫЧИЙ СИГНАЛ")
        else:
            signals.append("❌ MACD: МЕДВЕЖИЙ СИГНАЛ")

        # Тренд по SMA
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
            signals.append("🔼 ТРЕНД: ВОСХОДЯЩИЙ")
        else:
            signals.append("🔽 ТРЕНД: НИСХОДЯЩИЙ")

        return signals

    def create_stock_analysis_plot(self, symbol, data):
        """Создает график анализа акций"""
        df = self.calculate_technical_indicators(data)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # График цены и SMA
        ax1.plot(df.index, df['Close'], label='Цена', linewidth=2)
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.set_title(f'{symbol} - Цена и Скользящие Средние')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RSI
        ax2.plot(df.index, df['RSI'], label='RSI', color='orange', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Перекупленность')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Перепроданность')
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MACD
        ax3.plot(df.index, df['MACD'], label='MACD', linewidth=2)
        ax3.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=2)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохраняем в buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf


# Тестируем
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    data = yf.download('AAPL', period='3mo')
    signals = analyzer.generate_signals(data)
    print("Сигналы для AAPL:")
    for signal in signals:
        print(f"  {signal}")