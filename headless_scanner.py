import streamlit as st
import telebot
from telebot import types # Для кнопок
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading
import requests

# ==========================================
# 1. ГЛОБАЛЬНОЕ СОСТОЯНИЕ (С памятью)
# ==========================================
TG_TOKEN = "8407386703:AAEFkQ66ZOcGd7Ru41hrX34Bcb5BriNPuuQ"

# Инициализируем бота
bot = telebot.TeleBot(TG_TOKEN, threaded=False)

@st.cache_resource
def get_shared_state():
    return {
        "LENGTH_MAJOR": 200,
        "MAX_ATR_PCT": 5.0,
        "ADX_THRESH": 20,
        "AUTO_SCAN_INTERVAL": 60, 
        "IS_SCANNING": False,
        "STOP_SCAN": False,
        "SHOW_ONLY_NEW": True,
        "LAST_SCAN_TIME": "Никогда",
        "CHAT_ID": None,
        "NOTIFIED_TODAY": set(),
        "LAST_DATE": time.strftime("%Y-%m-%d")
    }

SETTINGS = get_shared_state()

# Текст помощи
HELP_TEXT = (
    "<b>🛠 Быстрые настройки:</b>\n"
    "Используйте кнопки меню внизу для изменения параметров."
)

# Функция для создания кнопок ГЛАВНОГО меню
def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn_scan = types.KeyboardButton('/scan 🚀')
    btn_stop = types.KeyboardButton('/stop 🛑')
    btn_stat = types.KeyboardButton('/status 📊')
    btn_mode = types.KeyboardButton('/mode 🔄')
    btn_atr = types.KeyboardButton('/atr_menu 📉') # Меню ATR
    btn_sma = types.KeyboardButton('/sma_menu 📈') # Меню SMA
    markup.add(btn_scan, btn_stop, btn_stat, btn_mode, btn_atr, btn_sma)
    return markup

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
            if attempt == 2: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

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

def perform_scan(chat_id, is_manual=False):
    if SETTINGS["IS_SCANNING"]:
        try: bot.send_message(chat_id, "⚠️ Сканирование уже идет!")
        except: pass
        return
    
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    current_date = time.strftime("%Y-%m-%d")
    if SETTINGS["LAST_DATE"] != current_date:
        SETTINGS["NOTIFIED_TODAY"] = set()
        SETTINGS["LAST_DATE"] = current_date
    
    mode_txt = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
    header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"

    status_msg = None
    try:
        status_msg = bot.send_message(chat_id, 
            f"{header}\nРежим: {mode_txt}\nSMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n⏳ Подготовка...", 
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
    except: pass
    
    tickers = get_sp500_tickers()
    total_tickers = len(tickers)
    found_count = 0
    
    for i, t in enumerate(tickers):
        if SETTINGS["STOP_SCAN"]:
            try: bot.send_message(chat_id, "🛑 Сканирование остановлено.")
            except: pass
            SETTINGS["IS_SCANNING"] = False
            return
        
        if i % 25 == 0 and status_msg:
            try:
                progress_pct = int((i / total_tickers) * 100)
                bar = "▓" * (progress_pct // 10) + "░" * (10 - (progress_pct // 10))
                new_text = f"{header}\nРежим: {mode_txt}\nSMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n⏳ {i}/{total_tickers} ({progress_pct}%)\n[{bar}]"
                bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=new_text, parse_mode="HTML")
            except: pass 

        res = check_ticker(t)
        if res:
            if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]:
                continue
            
            SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
            found_count += 1
            icon = "🔥 NEW" if res['is_new'] else "🟢"
            msg = f"{icon} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
            try: bot.send_message(chat_id, msg, parse_mode="HTML")
            except: pass
    
    try:
        final_text = f"✅ <b>Завершено</b>. Найдено: {found_count}" if found_count > 0 else f"🏁 <b>Завершено</b>. Ничего не найдено."
        if status_msg:
            bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=final_text, parse_mode="HTML")
        else:
            bot.send_message(chat_id, final_text, parse_mode="HTML")
            
        bot.send_message(chat_id, HELP_TEXT, parse_mode="HTML", reply_markup=get_main_keyboard())
        
    except: pass
    
    SETTINGS["IS_SCANNING"] = False
    SETTINGS["LAST_SCAN_TIME"] = time.strftime("%H:%M:%S")

# ==========================================
# 3. ОБРАБОТЧИКИ КОМАНД
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    bot.send_message(message.chat.id, 
        "👋 <b>Vova S&P 500 Screener</b>\n"
        "Бот активен. Используйте кнопки внизу для управления.",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )

# --- НОВЫЕ ХЕНДЛЕРЫ ДЛЯ МЕНЮ ---
@bot.message_handler(func=lambda message: message.text.startswith('/atr_menu'))
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton('/set_atr 3.0'),
        types.KeyboardButton('/set_atr 5.0'),
        types.KeyboardButton('/set_atr 7.0'),
        types.KeyboardButton('🔙 Назад')
    )
    bot.send_message(message.chat.id, "📉 <b>Выберите Max ATR %:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text.startswith('/sma_menu'))
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton('/set_sma 100'),
        types.KeyboardButton('/set_sma 150'),
        types.KeyboardButton('/set_sma 200'),
        types.KeyboardButton('🔙 Назад')
    )
    bot.send_message(message.chat.id, "📈 <b>Выберите SMA Period:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text.startswith('/mode'))
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    markup.add(
        types.KeyboardButton('Только НОВЫЕ 🔥'),
        types.KeyboardButton('ВСЕ активные 🟢'),
        types.KeyboardButton('🔙 Назад')
    )
    current_mode = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
    bot.send_message(message.chat.id, f"🔄 <b>Выберите режим сканирования:</b>\nТекущий: {current_mode}", parse_mode="HTML", reply_markup=markup)

# --- ОБРАБОТКА ВЫБОРА РЕЖИМА ---
@bot.message_handler(func=lambda message: message.text == 'Только НОВЫЕ 🔥')
def set_mode_new(message):
    SETTINGS["SHOW_ONLY_NEW"] = True
    bot.reply_to(message, "✅ Режим: <b>Только НОВЫЕ</b> (вход сегодня)", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda message: message.text == 'ВСЕ активные 🟢')
def set_mode_all(message):
    SETTINGS["SHOW_ONLY_NEW"] = False
    bot.reply_to(message, "✅ Режим: <b>ВСЕ активные</b> (любой зеленый тренд)", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda message: message.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

# --- ОБНОВЛЕННЫЕ СЕТТЕРЫ (ВОЗВРАЩАЮТ В ГЛАВНОЕ МЕНЮ) ---
@bot.message_handler(commands=['set_atr'])
def set_atr_val(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["MAX_ATR_PCT"] = val
        bot.reply_to(message, f"✅ ATR установлен: {val}%", reply_markup=get_main_keyboard())
    except: bot.reply_to(message, "❌ Пример: /set_atr 5.5")

@bot.message_handler(commands=['set_sma'])
def set_sma_val(message):
    try:
        val = int(message.text.split()[1])
        SETTINGS["LENGTH_MAJOR"] = val
        bot.reply_to(message, f"✅ SMA установлен: {val}", reply_markup=get_main_keyboard())
    except: bot.reply_to(message, "❌ Пример: /set_sma 200")

# --- ОСНОВНЫЕ КОМАНДЫ ---
@bot.message_handler(func=lambda message: message.text.startswith('/scan'))
def manual_scan(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    threading.Thread(target=perform_scan, args=(message.chat.id, True)).start()

@bot.message_handler(func=lambda message: message.text.startswith('/stop'))
def stop_scan(message):
    if SETTINGS["IS_SCANNING"]:
        SETTINGS["STOP_SCAN"] = True
        bot.reply_to(message, "🛑 Останавливаю...")
    else:
        bot.reply_to(message, "⚠️ Нет активного сканирования.", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda message: message.text.startswith('/status'))
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все"
    notified_count = len(SETTINGS["NOTIFIED_TODAY"])
    bot.reply_to(message, f"⚙️ <b>Настройки:</b>\nРежим: {mode}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nНайдено сегодня: {notified_count}\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML", reply_markup=get_main_keyboard())

# ==========================================
# 4. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=20, long_polling_timeout=10)
        except: time.sleep(5)

def start_scheduler():
    while True:
        time.sleep(60)
        if SETTINGS["CHAT_ID"]: perform_scan(SETTINGS["CHAT_ID"], False)
        time.sleep(3600) 

@st.cache_resource
def run_background_services():
    t1 = threading.Thread(target=start_polling, daemon=True)
    t1.start()
    t2 = threading.Thread(target=start_scheduler, daemon=True)
    t2.start()
    return True

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
st.title("🤖 Vova Bot Server")
run_background_services()
st.success("✅ Сервер активен! Кнопки добавлены.")
st.write(f"Отправлено сигналов сегодня: {len(SETTINGS['NOTIFIED_TODAY'])}")
st.metric("Последний скан", SETTINGS["LAST_SCAN_TIME"])

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300000, key="ref")
