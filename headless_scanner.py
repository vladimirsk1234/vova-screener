import streamlit as st
import telebot
from telebot import types
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading
import requests
import os
import json
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ==========================================
# 1. НАСТРОЙКИ И АВТОРИЗАЦИЯ
# ==========================================

try:
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
    ADMIN_ID = int(st.secrets.get("ADMIN_ID", 0))
    TG_TOKEN = st.secrets["TG_TOKEN"]
except:
    GITHUB_USERS_URL = ""
    ADMIN_ID = 0
    TG_TOKEN = ""

if not TG_TOKEN:
    st.error("❌ Токен не найден в Secrets!")
    st.stop()

# Сессия для yfinance (защита от блокировок)
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

YF_SESSION = get_session()

def fetch_approved_ids():
    """Загружает список ID из GitHub Raw URL"""
    ids = set()
    if ADMIN_ID != 0: ids.add(ADMIN_ID)
    if not GITHUB_USERS_URL: return ids
    try:
        response = requests.get(GITHUB_USERS_URL, timeout=10)
        if response.status_code == 200:
            for line in response.text.splitlines():
                line = line.strip()
                if line.isdigit(): ids.add(int(line))
    except Exception as e: print(f"GitHub fetch error: {e}")
    return ids

# Инициализация бота
bot = telebot.TeleBot(TG_TOKEN, threaded=False)

@st.cache_resource
def get_shared_state():
    return {
        "LENGTH_MAJOR": 200, 
        "MAX_ATR_PCT": 5.0, 
        "ADX_THRESH": 20,   
        "AUTO_SCAN_INTERVAL": 3600, 
        "IS_SCANNING": False, 
        "STOP_SCAN": False,
        "SHOW_ONLY_NEW": True, 
        "LAST_SCAN_TIME": "Никогда",
        "CHAT_IDS": set(), # Храним список всех активных чатов
        "APPROVED_IDS": fetch_approved_ids(), 
        "NOTIFIED_TODAY": set(), 
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0 
    }

SETTINGS = get_shared_state()

def is_authorized(user_id):
    if ADMIN_ID != 0 and user_id == ADMIN_ID: return True
    return user_id in SETTINGS["APPROVED_IDS"]

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2, one_time_keyboard=False)
    markup.row(types.KeyboardButton('Scan 🚀'), types.KeyboardButton('Stop 🛑'))
    markup.row(types.KeyboardButton('Status 📊'), types.KeyboardButton('Mode 🔄'))
    markup.row(types.KeyboardButton('ATR 📉'), types.KeyboardButton('SMA 📈'), types.KeyboardButton('Time 🕒'))
    return markup

def get_local_now():
    return datetime.utcnow() + timedelta(hours=SETTINGS["TIMEZONE_OFFSET"])

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
            return [str(t).replace('.', '-').strip() for t in table[0]['Symbol'].tolist()]
        except:
            time.sleep(2)
            if attempt == 2: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "ARES"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    """Логика из вашей последней рабочей версии"""
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, session=YF_SESSION)
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

        # Structure Logic
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
    sender_bot = telebot.TeleBot(TG_TOKEN)

    if SETTINGS["IS_SCANNING"]:
        try: sender_bot.send_message(chat_id, "⚠️ Сканирование уже идет!")
        except: pass
        return
    
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    try:
        local_now = get_local_now()
        current_date_str = local_now.strftime("%Y-%m-%d")
        if SETTINGS["LAST_DATE"] != current_date_str:
            SETTINGS["NOTIFIED_TODAY"] = set()
            SETTINGS["LAST_DATE"] = current_date_str
        
        mode_txt = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
        header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"
        tickers = get_sp500_tickers()
        
        start_text = (f"{header}\nРежим: {mode_txt}\n"
                      f"SMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
                      f"⏳ <b>Сканирование началось...</b>")
        sender_bot.send_message(chat_id, start_text, parse_mode="HTML")
        
        found_count = 0
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]: 
                sender_bot.send_message(chat_id, "🛑 Сканирование остановлено.")
                break
            
            # Анти-блок Yahoo (пауза каждые 15 акций)
            if i > 0 and i % 15 == 0: time.sleep(1)
            
            res = check_ticker(t)
            if res:
                if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]: continue
                SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
                found_count += 1
                icon = "🔥 NEW" if res['is_new'] else "🟢"
                msg = f"{icon} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
                
                # Отправка всем одобренным, если это авто-скан, или только инициатору, если ручной
                targets = [chat_id] if is_manual else list(SETTINGS["CHAT_IDS"])
                for target in targets:
                    if is_authorized(target):
                        try: sender_bot.send_message(target, msg, parse_mode="HTML")
                        except: pass
        
        final_text = f"✅ <b>Завершено</b>. Найдено: {found_count}" if found_count > 0 else "🏁 <b>Завершено</b>. Ничего не найдено."
        sender_bot.send_message(chat_id, final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            
    except Exception as e:
        sender_bot.send_message(chat_id, f"❌ Ошибка: {e}")
    finally:
        SETTINGS["IS_SCANNING"] = False
        SETTINGS["LAST_SCAN_TIME"] = get_local_now().strftime("%H:%M:%S")

# ==========================================
# 3. ОБРАБОТЧИКИ ТЕЛЕГРАМ
# ==========================================

@bot.message_handler(func=lambda m: not is_authorized(m.from_user.id))
def unauthorized_access(message):
    bot.send_message(message.chat.id, 
        f"⛔ <b>Доступ ограничен.</b>\nID: <code>{message.from_user.id}</code>", parse_mode="HTML")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_IDS"].add(message.chat.id)
    bot.send_message(message.chat.id, "👋 <b>Vova Bot Server Active</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['reload'])
def reload_users(message):
    if message.from_user.id != ADMIN_ID: return
    SETTINGS["APPROVED_IDS"] = fetch_approved_ids()
    bot.send_message(ADMIN_ID, f"✅ Список обновлен из GitHub ({len(SETTINGS['APPROVED_IDS'])} чел.)")

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀')
def manual_scan(message):
    threading.Thread(target=perform_scan, args=(message.chat.id, True)).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑')
def stop_scan(message):
    SETTINGS["STOP_SCAN"] = True
    bot.reply_to(message, "🛑 Останавливаю...")

@bot.message_handler(func=lambda m: m.text == 'ATR 📉')
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add('3.0 %', '5.0 %', '7.0 %', '10.0 %', '🔙 Назад')
    bot.send_message(message.chat.id, "📉 <b>Выберите Max ATR %:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'SMA 📈')
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add('100', '150', '200', '🔙 Назад')
    bot.send_message(message.chat.id, "📈 <b>Выберите SMA Period:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Status 📊')
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все"
    bot.reply_to(message, f"⚙️ <b>Статус:</b>\nРежим: {mode}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML")

@bot.message_handler(func=lambda m: m.text == 'Mode 🔄')
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    markup.add('Только НОВЫЕ 🔥', 'ВСЕ активные 🟢', '🔙 Назад')
    bot.send_message(message.chat.id, "🔄 <b>Выберите режим:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Только НОВЫЕ 🔥')
def set_mode_new(message):
    SETTINGS["SHOW_ONLY_NEW"] = True
    bot.reply_to(message, "✅ Режим: <b>Только НОВЫЕ</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'ВСЕ активные 🟢')
def set_mode_all(message):
    SETTINGS["SHOW_ONLY_NEW"] = False
    bot.reply_to(message, "✅ Режим: <b>ВСЕ активные</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: '%' in m.text or m.text.isdigit())
def handle_values(message):
    if '%' in message.text:
        SETTINGS["MAX_ATR_PCT"] = float(message.text.replace(' %',''))
        bot.reply_to(message, f"✅ ATR: {SETTINGS['MAX_ATR_PCT']}%", reply_markup=get_main_keyboard())
    elif message.text.isdigit():
        SETTINGS["LENGTH_MAJOR"] = int(message.text)
        bot.reply_to(message, f"✅ SMA: {SETTINGS['LENGTH_MAJOR']}", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'Time 🕒')
def check_time(message):
    local_time = get_local_now().strftime("%H:%M")
    bot.reply_to(message, f"🕒 Ваше время: <b>{local_time}</b>", parse_mode="HTML")

# ==========================================
# 4. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=20)
        except: time.sleep(5)

def start_scheduler():
    while True:
        time.sleep(60)
        # Рассылка всем активным чатам раз в час
        for chat_id in list(SETTINGS["CHAT_IDS"]):
            if is_authorized(chat_id):
                perform_scan(chat_id, False)
        time.sleep(3600)

@st.cache_resource
def run_background_services():
    threading.Thread(target=start_polling, daemon=True).start()
    threading.Thread(target=start_scheduler, daemon=True).start()
    return True

# ==========================================
# 5. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.title("🤖 Vova Bot Server")
run_background_services()
st.success("✅ Сервер активен. Одобрено пользователей: " + str(len(SETTINGS['APPROVED_IDS'])))
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300000, key="ref")
