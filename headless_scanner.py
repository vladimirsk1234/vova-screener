import streamlit as st
import telebot
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading
import requests

# ==========================================
# 1. НАСТРОЙКИ (Ваши данные)
# ==========================================
# Лучше хранить токены в st.secrets для безопасности, но для простоты оставим здесь
TG_TOKEN = "8407386703:AAEFkQ66ZOcGd7Ru41hrX34Bcb5BriNPuuQ"
# Chat ID бот запомнит сам после /start

# Глобальные настройки
SETTINGS = {
    "LENGTH_MAJOR": 200,
    "MAX_ATR_PCT": 5.0,
    "ADX_THRESH": 20,
    "AUTO_SCAN_INTERVAL": 60, # минут
    "IS_SCANNING": False,
    "STOP_SCAN": False,
    "SHOW_ONLY_NEW": True,
    "LAST_SCAN_TIME": "Никогда",
    "CHAT_ID": None # Будет сохранен после /start
}

# ==========================================
# 2. ФУНКЦИИ АНАЛИЗА
# ==========================================
def get_sp500_tickers():
    for attempt in range(3):
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            table = pd.read_html(io.StringIO(response.text))
            return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"] # Fallback

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 250: return None

        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR_Val'] = df['TR'].rolling(window=14).mean()
        df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
        
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        tr = pine_rma(df['TR'], 14); p_dm = pine_rma(df['+DM'], 14); m_dm = pine_rma(df['-DM'], 14)
        df['DI_Plus'] = 100 * (p_dm / tr); df['DI_Minus'] = 100 * (m_dm / tr)
        df['ADX'] = pine_rma(100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus']), 14)

        seqState = 0; seqHigh = df['High'].iloc[0]; seqLow = df['Low'].iloc[0]; crit = df['Low'].iloc[0]
        df_calc = df.iloc[-300:].copy()
        cl = df_calc['Close'].values; hi = df_calc['High'].values; lo = df_calc['Low'].values
        seq_states = []
        
        for i in range(len(df_calc)):
            c, h, l = cl[i], hi[i], lo[i]
            if i == 0: seq_states.append(0); continue
            pS = seq_states[-1]
            brk = (pS == 1 and c < crit) or (pS == -1 and c > crit)
            if brk:
                if pS == 1: seqState = -1; seqHigh = h; seqLow = l; crit = h
                else: seqState = 1; seqHigh = h; seqLow = l; crit = l
            else:
                if seqState == 1:
                    if h >= seqHigh: seqHigh = h
                    crit = l if h >= seqHigh else crit
                elif seqState == -1:
                    if l <= seqLow: seqLow = l
                    crit = h if l <= seqLow else crit
                else:
                    if c > seqHigh: seqState = 1; crit = l
                    elif c < seqLow: seqState = -1; crit = h
                    else: seqHigh = max(seqHigh, h); seqLow = min(seqLow, l)
            seq_states.append(seqState)

        last = df_calc.iloc[-1]; prev = df_calc.iloc[-2]
        if pd.isna(last['ADX']): return None
        
        seq_cur = seq_states[-1] == 1
        ma_cur = last['Close'] > last['SMA_Major']
        mom_cur = (last['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_cur and (last['DI_Plus'] > last['DI_Minus'])
        all_green_cur = seq_cur and ma_cur and mom_cur
        
        seq_prev = seq_states[-2] == 1
        ma_prev = prev['Close'] > prev['SMA_Major']
        mom_prev = (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_prev and (prev['DI_Plus'] > prev['DI_Minus'])
        all_green_prev = seq_prev and ma_prev and mom_prev
        
        pass_filters = (last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"])
        is_new_signal = all_green_cur and not all_green_prev

        if all_green_cur and pass_filters:
            if not SETTINGS["SHOW_ONLY_NEW"] or is_new_signal:
                return {'ticker': ticker, 'price': last['Close'], 'atr': last['ATR_Pct'], 'is_new': is_new_signal}
    except: return None
    return None

def perform_scan(chat_id):
    if SETTINGS["IS_SCANNING"]: return
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    bot = telebot.TeleBot(TG_TOKEN)
    try:
        bot.send_message(chat_id, "🚀 <b>Начинаю ежечасное сканирование...</b>", parse_mode="HTML")
    except: pass
    
    tickers = get_sp500_tickers()
    found_count = 0
    
    for i, t in enumerate(tickers):
        if SETTINGS["STOP_SCAN"]: break
        res = check_ticker(t)
        if res:
            found_count += 1
            icon = "🔥 NEW" if res['is_new'] else "🟢"
            msg = f"{icon} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
            try: bot.send_message(chat_id, msg, parse_mode="HTML")
            except: pass
    
    try:
        if found_count > 0:
            bot.send_message(chat_id, f"✅ Завершено. Найдено: {found_count}")
        else:
            # Можно отключить это сообщение, чтобы не спамить каждый час если пусто
            bot.send_message(chat_id, "🤷‍♂️ Новых сигналов нет.") 
    except: pass
    
    SETTINGS["IS_SCANNING"] = False
    SETTINGS["LAST_SCAN_TIME"] = time.strftime("%H:%M:%S")

# ==========================================
# 3. TELEGRAM БОТ (В ОТДЕЛЬНОМ ПОТОКЕ)
# ==========================================
def start_bot_polling():
    bot = telebot.TeleBot(TG_TOKEN)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        SETTINGS["CHAT_ID"] = message.chat.id
        bot.reply_to(message, "✅ <b>Бот активирован!</b>\nТеперь я буду присылать сюда сигналы каждый час.\nУбедитесь, что Streamlit вкладка открыта.", parse_mode="HTML")

    @bot.message_handler(commands=['scan'])
    def manual_scan(message):
        threading.Thread(target=perform_scan, args=(message.chat.id,)).start()

    @bot.message_handler(commands=['status'])
    def status(message):
        bot.send_message(message.chat.id, f"Последнее сканирование: {SETTINGS['LAST_SCAN_TIME']}")

    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except:
            time.sleep(5)

# ==========================================
# 4. ФОНОВЫЙ ТАЙМЕР (КАЖДЫЙ ЧАС)
# ==========================================
def hourly_scheduler():
    while True:
        time.sleep(60) # Проверяем каждую минуту, не прошел ли час
        # Простая логика: спим час и запускаем
        # В реальном коде лучше использовать schedule, но для простоты sleep
        if SETTINGS["CHAT_ID"]:
            perform_scan(SETTINGS["CHAT_ID"])
        time.sleep(3600) # Ждем 1 час (3600 сек)

# ==========================================
# 5. ИНТЕРФЕЙС STREAMLIT (ЧТОБЫ РАБОТАЛО В ОБЛАКЕ)
# ==========================================
st.title("🤖 Vova Telegram Bot Server")
st.write("Этот сервер должен быть запущен, чтобы бот работал.")

if "bot_started" not in st.session_state:
    st.session_state["bot_started"] = True
    
    # Запускаем бота в отдельном потоке
    t_bot = threading.Thread(target=start_bot_polling, daemon=True)
    t_bot.start()
    
    # Запускаем таймер в отдельном потоке
    t_schedule = threading.Thread(target=hourly_scheduler, daemon=True)
    t_schedule.start()
    
    st.success("Бот и планировщик запущены!")

st.metric("Последнее сканирование", SETTINGS["LAST_SCAN_TIME"])
st.write(f"Chat ID: {SETTINGS.get('CHAT_ID', 'Ожидание /start...')}")

# Хак для предотвращения засыпания (обновляет страницу раз в 5 минут)
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5 * 60 * 1000, key="refresh")
