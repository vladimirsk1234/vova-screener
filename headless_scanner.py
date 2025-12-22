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
from datetime import datetime, timedelta

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ==========================================
# 1. НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# ==========================================

try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
except:
    TG_TOKEN = os.environ.get("TG_TOKEN", "") 

if not TG_TOKEN:
    st.error("❌ **ОШИБКА:** Токен не найден! Добавьте его в Secrets на Streamlit Cloud.")
    st.stop()

try:
    bot = telebot.TeleBot(TG_TOKEN, threaded=True)
except Exception as e:
    st.error(f"❌ Ошибка бота: {e}")
    st.stop()

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
        "CHAT_ID": None,
        "NOTIFIED_TODAY": set(),
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0,
        "TICKER_LIMIT": 500 
    }

SETTINGS = get_shared_state()

# ГЛОБАЛЬНОЕ СОСТОЯНИЕ ПРОГРЕССА (Для отделения UI от логики)
PROGRESS = {
    "current": 0,
    "total": 0,
    "running": False,
    "msg_id": None,
    "chat_id": None,
    "header": ""
}

# --- МЕНЮ ---
def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3, one_time_keyboard=False)
    markup.row(types.KeyboardButton('Scan 🚀'), types.KeyboardButton('Stop 🛑'))
    markup.row(types.KeyboardButton('Status 📊'), types.KeyboardButton('Mode 🔄'))
    markup.row(types.KeyboardButton('ATR 📉'), types.KeyboardButton('SMA 📈'), types.KeyboardButton('Time 🕒'))
    return markup

# --- ВРЕМЯ ---
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
            return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 200: return None

        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        tr = pine_rma(df['TR'], 14); p_dm = pine_rma(df['+DM'], 14); m_dm = pine_rma(df['-DM'], 14)
        df['DI_Plus'] = 100 * (p_dm / tr); df['DI_Minus'] = 100 * (m_dm / tr)
        df['ADX'] = pine_rma(100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus']), 14)

        seqState = 0; seqHigh = df['High'].iloc[0]; seqLow = df['Low'].iloc[0]; crit = df['Low'].iloc[0]
        df_calc = df.iloc[-250:].copy()
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
        
        all_green_cur = (seq_states[-1] == 1) and (last['Close'] > last['SMA_Major']) and (last['ADX'] >= SETTINGS["ADX_THRESH"]) and (last['DI_Plus'] > last['DI_Minus'])
        all_green_prev = (seq_states[-2] == 1) and (prev['Close'] > prev['SMA_Major']) and (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and (prev['DI_Plus'] > prev['DI_Minus'])
        
        pass_filters = (last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"])
        is_new_signal = all_green_cur and not all_green_prev

        if all_green_cur and pass_filters:
            if not SETTINGS["SHOW_ONLY_NEW"] or is_new_signal:
                return {'ticker': ticker, 'price': last['Close'], 'atr': last['ATR_Pct'], 'is_new': is_new_signal}
    except: return None
    return None

# ==========================================
# 3. ПОТОК ОБНОВЛЕНИЯ ИНТЕРФЕЙСА (HEARTBEAT)
# ==========================================
def progress_updater():
    """Легкий поток, который только обновляет сообщение в Telegram раз в 4 секунды"""
    while PROGRESS["running"]:
        try:
            if PROGRESS["total"] > 0:
                pct = int((PROGRESS["current"] / PROGRESS["total"]) * 100)
                bar_filled = pct // 10
                bar_str = "▓" * bar_filled + "░" * (10 - bar_filled)
                
                text = (
                    f"{PROGRESS['header']}\n"
                    f"SMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
                    f"Прогресс: {PROGRESS['current']}/{PROGRESS['total']} ({pct}%)\n"
                    f"[{bar_str}]"
                )
                
                bot.edit_message_text(
                    chat_id=PROGRESS["chat_id"],
                    message_id=PROGRESS["msg_id"],
                    text=text,
                    parse_mode="HTML"
                )
        except Exception as e:
            # Если Telegram отклоняет правку (Flood Limit), просто ждем следующего цикла
            pass
        time.sleep(4)

def perform_scan(chat_id, is_manual=False):
    if SETTINGS["IS_SCANNING"]:
        try: bot.send_message(chat_id, "⚠️ Сканирование уже идет!")
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
        
        header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"
        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        # Создаем сообщение для прогресса
        status_msg = bot.send_message(chat_id, "⏳ Инициализация сканирования...", parse_mode="HTML")
        
        # Инициализируем глобальный прогресс и запускаем поток обновления UI
        PROGRESS.update({
            "current": 0,
            "total": total_tickers,
            "running": True,
            "msg_id": status_msg.message_id,
            "chat_id": chat_id,
            "header": header
        })
        
        ui_thread = threading.Thread(target=progress_updater, daemon=True)
        ui_thread.start()
        
        found_count = 0
        
        # ОСНОВНОЙ ЦИКЛ (Тяжелая работа)
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]:
                PROGRESS["running"] = False # Останавливаем UI поток
                try: bot.send_message(chat_id, "🛑 Сканирование остановлено.")
                except: pass
                break
            
            # Обновляем ТОЛЬКО счетчик (UI поток подхватит его сам)
            PROGRESS["current"] = i + 1

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
        
        # Останавливаем поток обновления UI перед финальным сообщением
        PROGRESS["running"] = False
        time.sleep(1) # Даем потоку завершиться
        
        # --- ФИНАЛ ---
        final_text = f"✅ <b>Завершено</b>. Найдено: {found_count}" if found_count > 0 else f"🏁 <b>Завершено</b>. Ничего не найдено."
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
        except:
            bot.send_message(chat_id, final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            
    except Exception as e:
        print(f"Global scan error: {e}")
    finally:
        PROGRESS["running"] = False
        SETTINGS["IS_SCANNING"] = False
        SETTINGS["LAST_SCAN_TIME"] = get_local_now().strftime("%H:%M:%S")

# ==========================================
# 4. ОБРАБОТЧИКИ КОМАНД
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    bot.send_message(message.chat.id, 
        "👋 <b>Vova S&P 500 Screener</b>\nБот активен. Нажмите кнопку меню ниже.",
        parse_mode="HTML", reply_markup=get_main_keyboard()
    )

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀' or m.text.startswith('/scan'))
def manual_scan(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    threading.Thread(target=perform_scan, args=(message.chat.id, True), daemon=True).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑' or m.text.startswith('/stop'))
def stop_scan(message):
    if SETTINGS["IS_SCANNING"]:
        SETTINGS["STOP_SCAN"] = True
        bot.reply_to(message, "🛑 Останавливаю...")
    else:
        bot.reply_to(message, "⚠️ Нет активного сканирования.")

@bot.message_handler(func=lambda m: m.text == 'Status 📊' or m.text.startswith('/status'))
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все"
    bot.reply_to(message, 
        f"⚙️ <b>Настройки:</b>\nРежим: {mode}\nSMA: {SETTINGS['LENGTH_MAJOR']}\n"
        f"Max ATR: {SETTINGS['MAX_ATR_PCT']}%\nНайдено сегодня: {len(SETTINGS['NOTIFIED_TODAY'])}\n"
        f"Посл. скан: {SETTINGS['LAST_SCAN_TIME']}", 
        parse_mode="HTML"
    )

@bot.message_handler(func=lambda m: m.text == 'Time 🕒')
def check_time(message):
    local_time = get_local_now().strftime("%H:%M")
    bot.reply_to(message, f"🕒 Ваше локальное время: <b>{local_time}</b> (UTC{SETTINGS['TIMEZONE_OFFSET']})", parse_mode="HTML")

@bot.message_handler(commands=['set_offset'])
def set_offset(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["TIMEZONE_OFFSET"] = val
        bot.reply_to(message, f"✅ Смещение UTC установлено: {val}")
    except:
        bot.reply_to(message, "❌ Ошибка. Пример: /set_offset -7")

# --- МЕНЮ ATR / SMA ---
@bot.message_handler(func=lambda m: m.text == 'ATR 📉')
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('3.0 %', '5.0 %', '7.0 %', '10.0 %', '🔙 Назад')
    bot.send_message(message.chat.id, "📉 Выберите Max ATR:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'SMA 📈')
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('100', '150', '200', '🔙 Назад')
    bot.send_message(message.chat.id, "📈 Выберите SMA Period:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Mode 🔄')
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('Только НОВЫЕ 🔥', 'ВСЕ активные 🟢', '🔙 Назад')
    bot.send_message(message.chat.id, "🔄 Режим отображения:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: '%' in m.text or m.text in ['100', '150', '200', 'Только НОВЫЕ 🔥', 'ВСЕ активные 🟢'])
def handle_settings(message):
    if '%' in message.text:
        SETTINGS["MAX_ATR_PCT"] = float(message.text.replace(' %',''))
        bot.send_message(message.chat.id, f"✅ ATR: {SETTINGS['MAX_ATR_PCT']}%", reply_markup=get_main_keyboard())
    elif message.text.isdigit():
        SETTINGS["LENGTH_MAJOR"] = int(message.text)
        bot.send_message(message.chat.id, f"✅ SMA: {SETTINGS['LENGTH_MAJOR']}", reply_markup=get_main_keyboard())
    elif 'НОВЫЕ' in message.text:
        SETTINGS["SHOW_ONLY_NEW"] = True
        bot.send_message(message.chat.id, "✅ Только новые", reply_markup=get_main_keyboard())
    elif 'ВСЕ' in message.text:
        SETTINGS["SHOW_ONLY_NEW"] = False
        bot.send_message(message.chat.id, "✅ Все активные", reply_markup=get_main_keyboard())

# ==========================================
# 5. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=30, long_polling_timeout=20)
        except: time.sleep(5)

def start_scheduler():
    while True:
        time.sleep(3600)
        if SETTINGS["CHAT_ID"] and not SETTINGS["IS_SCANNING"]:
            perform_scan(SETTINGS["CHAT_ID"], False)

@st.cache_resource
def run_background_services():
    t1 = threading.Thread(target=start_polling, daemon=True)
    t1.start()
    t2 = threading.Thread(target=start_scheduler, daemon=True)
    t2.start()
    return True

# ==========================================
# 6. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.title("🤖 Vova Bot Server")

st.image("https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1000&auto=format&fit=crop", 
         use_container_width=True)

run_background_services()
st.success("✅ Сервер активен! Бот работает в фоновом режиме.")
st.write(f"Найдено сигналов за сегодня: {len(SETTINGS['NOTIFIED_TODAY'])}")
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60000, key="ref")
