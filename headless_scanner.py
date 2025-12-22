import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import os

# ==========================================
# 1. НАСТРОЙКИ TELEGRAM (Впишите данные сюда)
# ==========================================
TG_TOKEN = "8407386703:AAEFkQ66ZOcGd7Ru41hrX34Bcb5BriNPuuQ"   # Пример: "123456:ABC-DEF..."
TG_CHAT_ID = "1335722880"    # Пример: "12345678"

# Остальные настройки
LENGTH_MAJOR = 200
MAX_ATR_PCT = 10.0
MIN_MCAP = 10.0
ADX_THRESH = 20

def send_telegram(message):
    # Проверка, вписал ли пользователь токен
    if "ВСТАВЬТЕ" in TG_TOKEN or "ВСТАВЬТЕ" in TG_CHAT_ID:
        print("❌ ОШИБКА: Вы не вписали Token или Chat ID в код файла!")
        return False
        
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True
        else:
            print(f"Ошибка Telegram: {response.text}")
            return False
    except Exception as e:
        print(f"Ошибка сети: {e}")
        return False

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))
        tickers = table[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        # Запасной список, если википедия не грузится
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "JPM", "BAC", "CSCO"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    try:
        # Качаем данные
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if len(df) < 250: return None

        # --- РАСЧЕТЫ ---
        df['SMA_Major'] = df['Close'].rolling(window=LENGTH_MAJOR).mean()
        
        # ATR
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR_Val'] = df['TR'].rolling(window=14).mean()
        df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
        
        # ADX
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        tr_smooth = pine_rma(df['TR'], 14)
        plus_dm = pine_rma(df['+DM'], 14)
        minus_dm = pine_rma(df['-DM'], 14)
        df['DI_Plus'] = 100 * (plus_dm / tr_smooth)
        df['DI_Minus'] = 100 * (minus_dm / tr_smooth)
        dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = pine_rma(dx, 14)

        # SEQUENCE
        seqState = 0; seqHigh = df['High'].iloc[0]; seqLow = df['Low'].iloc[0]; criticalLevel = df['Low'].iloc[0]
        # Для ускорения считаем только конец истории
        df_calc = df.iloc[-300:].copy()
        
        closes = df_calc['Close'].values; highs = df_calc['High'].values; lows = df_calc['Low'].values
        seq_states = []
        
        for i in range(len(df_calc)):
            c, h, l = closes[i], highs[i], lows[i]
            if i == 0: seq_states.append(0); continue
            
            pS = seq_states[-1]
            brk = (pS == 1 and c < criticalLevel) or (pS == -1 and c > criticalLevel)
            
            if brk:
                if pS == 1: seqState = -1; seqHigh = h; seqLow = l; criticalLevel = h
                else: seqState = 1; seqHigh = h; seqLow = l; criticalLevel = l
            else:
                if seqState == 1:
                    if h >= seqHigh: seqHigh = h
                    criticalLevel = l if h >= seqHigh else criticalLevel
                elif seqState == -1:
                    if l <= seqLow: seqLow = l
                    criticalLevel = h if l <= seqLow else criticalLevel
                else:
                    if c > seqHigh: seqState = 1; criticalLevel = l
                    elif c < seqLow: seqState = -1; criticalLevel = h
                    else: seqHigh = max(seqHigh, h); seqLow = min(seqLow, l)
            seq_states.append(seqState)

        # --- ПРОВЕРКА ---
        last = df_calc.iloc[-1]
        prev = df_calc.iloc[-2]
        
        if pd.isna(last['ADX']): return None
        
        # Логика сигналов
        seq_cur = seq_states[-1] == 1
        ma_cur = last['Close'] > last['SMA_Major']
        mom_cur = (last['ADX'] >= ADX_THRESH) and seq_cur and (last['DI_Plus'] > last['DI_Minus'])
        all_green_cur = seq_cur and ma_cur and mom_cur
        
        seq_prev = seq_states[-2] == 1
        ma_prev = prev['Close'] > prev['SMA_Major']
        mom_prev = (prev['ADX'] >= ADX_THRESH) and seq_prev and (prev['DI_Plus'] > prev['DI_Minus'])
        all_green_prev = seq_prev and ma_prev and mom_prev
        
        # Проверка фильтров
        try: mcap = yf.Ticker(ticker).fast_info.market_cap / 1_000_000_000
        except: mcap = 100 
            
        pass_filters = (last['ATR_Pct'] <= MAX_ATR_PCT) and (mcap >= MIN_MCAP)
        
        # ВОЗВРАЩАЕМ РЕЗУЛЬТАТ ТОЛЬКО ЕСЛИ СИГНАЛ НОВЫЙ И ФИЛЬТРЫ ПРОЙДЕНЫ
        if all_green_cur and not all_green_prev and pass_filters:
            return {
                'ticker': ticker,
                'price': last['Close'],
                'atr': last['ATR_Pct']
            }
            
    except Exception as e:
        return None
    return None

def main():
    print(f"🚀 Запуск сканера... (Token: {'OK' if 'ВСТАВЬТЕ' not in TG_TOKEN else 'НЕ ЗАДАН'})")
    
    # Тестовая отправка сообщения при запуске
    test_sent = send_telegram("🤖 Бот-сканер запущен на компьютере!")
    if test_sent:
        print("✅ Тестовое сообщение отправлено успешно.")
    else:
        print("❌ Не удалось отправить тест. Проверьте Token/ID.")

    tickers = get_sp500_tickers()
    # tickers = tickers[:20] # Раскомментируйте для быстрого теста на 20 тикерах
    
    print(f"Сканирую {len(tickers)} тикеров...")
    
    found_count = 0
    for i, t in enumerate(tickers):
        res = check_ticker(t)
        if res:
            found_count += 1
            msg = f"🚀 <b>NEW SIGNAL: {res['ticker']}</b>\nPrice: ${res['price']:.2f}\nATR: {res['atr']:.2f}%"
            sent = send_telegram(msg)
            if sent:
                print(f"[{i+1}/{len(tickers)}] 🟢 {t}: Сигнал найден и отправлен!")
            else:
                print(f"[{i+1}/{len(tickers)}] 🟡 {t}: Сигнал найден, но ОШИБКА отправки.")
        
        # Вывод прогресса в одну строку, чтобы не спамить в консоль
        if i % 5 == 0:
            print(f"Обработано {i}/{len(tickers)}...", end='\r')
    
    print(f"\nГотово. Всего найдено: {found_count}")

if __name__ == "__main__":
    main()

