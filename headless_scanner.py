import telebot
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading

# ==========================================
# 1. НАСТРОЙКИ (Ваши данные)
# ==========================================
TG_TOKEN = "8407386703:AAEFkQ66ZOcGd7Ru41hrX34Bcb5BriNPuuQ"
# Chat ID здесь больше не нужен жестко, бот сам узнает его, когда вы напишете /start

# Инициализация бота
bot = telebot.TeleBot(TG_TOKEN)

# Глобальные переменные для настроек (чтобы их можно было менять через Telegram)
SETTINGS = {
    "LENGTH_MAJOR": 200,
    "MAX_ATR_PCT": 10.0,
    "MIN_MCAP": 10.0,
    "ADX_THRESH": 20,
    "AUTO_SCAN_INTERVAL": 60, # минут (0 = выкл)
    "IS_SCANNING": False
}

# ==========================================
# 2. ФУНКЦИИ АНАЛИЗА (Те же самые)
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))
        tickers = table[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "JPM", "BAC"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 250: return None

        # SMA
        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
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

        # CHECK LAST BAR
        last = df_calc.iloc[-1]
        prev = df_calc.iloc[-2]
        
        if pd.isna(last['ADX']): return None
        
        # Logic
        seq_cur = seq_states[-1] == 1
        ma_cur = last['Close'] > last['SMA_Major']
        mom_cur = (last['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_cur and (last['DI_Plus'] > last['DI_Minus'])
        all_green_cur = seq_cur and ma_cur and mom_cur
        
        seq_prev = seq_states[-2] == 1
        ma_prev = prev['Close'] > prev['SMA_Major']
        mom_prev = (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_prev and (prev['DI_Plus'] > prev['DI_Minus'])
        all_green_prev = seq_prev and ma_prev and mom_prev
        
        # Filters
        try: mcap = yf.Ticker(ticker).fast_info.market_cap / 1_000_000_000
        except: mcap = 100 
            
        pass_filters = (last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"]) and (mcap >= SETTINGS["MIN_MCAP"])
        
        if all_green_cur and not all_green_prev and pass_filters:
            return {
                'ticker': ticker,
                'price': last['Close'],
                'atr': last['ATR_Pct']
            }
    except: return None
    return None

def perform_scan(chat_id):
    if SETTINGS["IS_SCANNING"]:
        bot.send_message(chat_id, "⚠️ Сканирование уже идет! Подождите.")
        return

    SETTINGS["IS_SCANNING"] = True
    bot.send_message(chat_id, "🚀 <b>Начинаю сканирование рынка...</b>\nЭто займет 1-2 минуты.", parse_mode="HTML")
    
    tickers = get_sp500_tickers()
    found_count = 0
    
    for i, t in enumerate(tickers):
        res = check_ticker(t)
        if res:
            found_count += 1
            msg = f"🔥 <b>NEW SIGNAL: {res['ticker']}</b>\nPrice: ${res['price']:.2f}\nATR: {res['atr']:.2f}%"
            bot.send_message(chat_id, msg, parse_mode="HTML")
    
    if found_count == 0:
        bot.send_message(chat_id, "🤷‍♂️ Новых сигналов пока нет.")
    else:
        bot.send_message(chat_id, f"✅ Сканирование завершено. Найдено: {found_count}")
    
    SETTINGS["IS_SCANNING"] = False

# ==========================================
# 3. TELEGRAM КОМАНДЫ
# ==========================================

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 
        "👋 Привет! Я Vova Screener Bot.\n\n"
        "<b>Команды:</b>\n"
        "/scan - Запустить поиск вручную\n"
        "/status - Текущие настройки\n"
        "/set_atr 10 - Установить Max ATR %\n"
        "/set_mcap 10 - Установить Min Market Cap (B$)\n"
        "/set_sma 200 - Установить период SMA",
        parse_mode="HTML"
    )

@bot.message_handler(commands=['scan'])
def command_scan(message):
    # Запускаем в отдельном потоке, чтобы бот не завис
    threading.Thread(target=perform_scan, args=(message.chat.id,)).start()

@bot.message_handler(commands=['status'])
def command_status(message):
    msg = (
        f"⚙️ <b>Текущие настройки:</b>\n"
        f"• SMA Period: {SETTINGS['LENGTH_MAJOR']}\n"
        f"• Max ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
        f"• Min M.Cap: ${SETTINGS['MIN_MCAP']}B\n"
        f"• Min ADX: {SETTINGS['ADX_THRESH']}"
    )
    bot.send_message(message.chat.id, msg, parse_mode="HTML")

@bot.message_handler(commands=['set_atr'])
def set_atr(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["MAX_ATR_PCT"] = val
        bot.reply_to(message, f"✅ Max ATR установлен на {val}%")
    except:
        bot.reply_to(message, "❌ Ошибка. Пример: /set_atr 5.5")

@bot.message_handler(commands=['set_mcap'])
def set_mcap(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["MIN_MCAP"] = val
        bot.reply_to(message, f"✅ Min Market Cap установлен на ${val}B")
    except:
        bot.reply_to(message, "❌ Ошибка. Пример: /set_mcap 20")

@bot.message_handler(commands=['set_sma'])
def set_sma(message):
    try:
        val = int(message.text.split()[1])
        SETTINGS["LENGTH_MAJOR"] = val
        bot.reply_to(message, f"✅ SMA Period установлен на {val}")
    except:
        bot.reply_to(message, "❌ Ошибка. Пример: /set_sma 200")

# ==========================================
# 4. ЗАПУСК БОТА
# ==========================================
if __name__ == "__main__":
    import requests # Импорт нужен внутри для функций
    print("🤖 Бот запущен! Пишите /scan в Telegram.")
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"Ошибка бота: {e}")
        time.sleep(5)
