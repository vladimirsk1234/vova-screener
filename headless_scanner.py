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
    st.error("❌ **ОШИБКА:** Токен не найден! Добавьте его в Secrets.")
    st.stop()

try:
    bot = telebot.TeleBot(TG_TOKEN, threaded=False)
except Exception as e:
    st.error(f"❌ Ошибка бота: {e}")
    st.stop()

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
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0 # Установлено UTC-7
    }

SETTINGS = get_shared_state()

# --- МЕНЮ ---
def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2, one_time_keyboard=False)
    # 1 ряд
    markup.row(types.KeyboardButton('Scan 🚀'), types.KeyboardButton('Stop 🛑'))
    # 2 ряд
    markup.row(types.KeyboardButton('Status 📊'), types.KeyboardButton('Mode 🔄'))
    # 3 ряд (Настройки) - Кнопка Limit удалена
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
    # Создаем отдельного бота для отправки, чтобы не блокировать потоки
    sender_bot = telebot.TeleBot(TG_TOKEN)

    if SETTINGS["IS_SCANNING"]:
        try: sender_bot.send_message(chat_id, "⚠️ Сканирование уже идет!", reply_markup=get_main_keyboard())
        except: pass
        return
    
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    try:
        # Сброс дневного лимита при смене даты
        local_now = get_local_now()
        current_date_str = local_now.strftime("%Y-%m-%d")
        
        if SETTINGS["LAST_DATE"] != current_date_str:
            SETTINGS["NOTIFIED_TODAY"] = set()
            SETTINGS["LAST_DATE"] = current_date_str
        
        mode_txt = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
        header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"

        # Получаем тикеры (ВСЕГДА ПОЛНЫЙ СКАН)
        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        # Отправляем простое СТАРТОВОЕ сообщение (без прогресса)
        try:
            start_text = (
                f"{header}\nРежим: {mode_txt}\n"
                f"SMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
                f"Всего акций: {total_tickers}\n"
                f"⏳ <b>Сканирование началось...</b>"
            )
            sender_bot.send_message(chat_id, start_text, parse_mode="HTML", reply_markup=get_main_keyboard())
        except Exception as e:
            print(f"Start msg error: {e}")
        
        found_count = 0
        
        # Основной цикл
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]:
                try: sender_bot.send_message(chat_id, "🛑 Сканирование остановлено.", reply_markup=get_main_keyboard())
                except: pass
                break
            
            # Проверка акции
            res = check_ticker(t)
            if res:
                if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]:
                    continue
                
                SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
                found_count += 1
                icon = "🔥 NEW" if res['is_new'] else "🟢"
                msg = f"{icon} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
                try: sender_bot.send_message(chat_id, msg, parse_mode="HTML", reply_markup=get_main_keyboard())
                except: pass
        
        # --- ФИНАЛ ---
        try:
            if found_count == 0:
                final_text = f"🏁 <b>Сканирование завершено</b>\n🤷‍♂️ Ничего не найдено."
            else:
                final_text = f"✅ <b>Сканирование завершено</b>\nНайдено сигналов: {found_count}"
            
            sender_bot.send_message(chat_id, final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            
        except: pass
        
    except Exception as e:
        print(f"Scan error: {e}")
        try: sender_bot.send_message(chat_id, f"❌ Ошибка сканирования: {e}", reply_markup=get_main_keyboard())
        except: pass
    
    finally:
        SETTINGS["IS_SCANNING"] = False
        SETTINGS["LAST_SCAN_TIME"] = get_local_now().strftime("%H:%M:%S")

# ==========================================
# 3. ОБРАБОТЧИКИ КОМАНД
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    bot.send_message(message.chat.id, 
        "👋 <b>Vova S&P 500 Screener</b>\n"
        "Бот активен. Нажмите кнопку меню ниже.",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )

# --- МЕНЮ ATR ---
@bot.message_handler(func=lambda m: m.text == 'ATR 📉' or m.text.startswith('/atr_menu'))
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton('3.0 %'),
        types.KeyboardButton('5.0 %'),
        types.KeyboardButton('7.0 %'),
        types.KeyboardButton('10.0 %'),
        types.KeyboardButton('🔙 Назад')
    )
    bot.send_message(message.chat.id, "📉 <b>Выберите Max ATR %:</b>", parse_mode="HTML", reply_markup=markup)

# --- МЕНЮ SMA ---
@bot.message_handler(func=lambda m: m.text == 'SMA 📈' or m.text.startswith('/sma_menu'))
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton('100'),
        types.KeyboardButton('150'),
        types.KeyboardButton('200'),
        types.KeyboardButton('🔙 Назад')
    )
    bot.send_message(message.chat.id, "📈 <b>Выберите SMA Period:</b>", parse_mode="HTML", reply_markup=markup)

# --- МЕНЮ ВРЕМЕНИ ---
@bot.message_handler(func=lambda m: m.text == 'Time 🕒' or m.text.startswith('/time'))
def check_time(message):
    server_time = datetime.utcnow().strftime("%H:%M")
    local_time = get_local_now().strftime("%H:%M")
    offset = SETTINGS["TIMEZONE_OFFSET"]
    off_str = f"+{offset}" if offset >= 0 else f"{offset}"
    
    bot.reply_to(message, 
        f"🕒 <b>Системное время:</b>\n"
        f"☁️ Сервер (UTC): {server_time}\n"
        f"🏠 Ваше (UTC{off_str}): <b>{local_time}</b>\n\n"
        f"Чтобы изменить часовой пояс, напишите:\n<code>/set_offset -7</code>", 
        parse_mode="HTML", reply_markup=get_main_keyboard()
    )

@bot.message_handler(func=lambda message: message.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

# --- УСТАНОВКА ЗНАЧЕНИЙ ---
@bot.message_handler(func=lambda m: '%' in m.text and m.text.replace(' %','').replace('.','').isdigit())
def set_atr_text(message):
    try:
        val = float(message.text.replace(' %',''))
        SETTINGS["MAX_ATR_PCT"] = val
        bot.reply_to(message, f"✅ ATR установлен: {val}%", reply_markup=get_main_keyboard())
    except: 
        bot.reply_to(message, "❌ Ошибка", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text in ['100', '150', '200'])
def set_sma_text(message):
    try:
        val = int(message.text)
        SETTINGS["LENGTH_MAJOR"] = val
        bot.reply_to(message, f"✅ SMA установлен: {val}", reply_markup=get_main_keyboard())
    except:
        bot.reply_to(message, "❌ Ошибка", reply_markup=get_main_keyboard())

# --- НАСТРОЙКА ЧАСОВОГО ПОЯСА ---
@bot.message_handler(commands=['set_offset'])
def set_offset(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["TIMEZONE_OFFSET"] = val
        curr_time = get_local_now().strftime("%H:%M")
        bot.reply_to(message, f"✅ Смещение UTC: {val}\n⏰ Текущее время: {curr_time}", reply_markup=get_main_keyboard())
    except:
        bot.reply_to(message, "❌ Ошибка. Пример: <code>/set_offset -7</code>", parse_mode="HTML")

# --- ОСНОВНЫЕ КНОПКИ ---
@bot.message_handler(func=lambda m: m.text == 'Scan 🚀' or m.text.startswith('/scan'))
def manual_scan(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    threading.Thread(target=perform_scan, args=(message.chat.id, True)).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑' or m.text.startswith('/stop'))
def stop_scan(message):
    if SETTINGS["IS_SCANNING"]:
        SETTINGS["STOP_SCAN"] = True
        bot.reply_to(message, "🛑 Останавливаю...", reply_markup=get_main_keyboard())
    else:
        bot.reply_to(message, "⚠️ Нет активного сканирования.", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'Status 📊' or m.text.startswith('/status'))
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все"
    notified_count = len(SETTINGS["NOTIFIED_TODAY"])
    offset = SETTINGS["TIMEZONE_OFFSET"]
    bot.reply_to(message, f"⚙️ <b>Настройки:</b>\nРежим: {mode}\nЧасовой пояс: {offset}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nНайдено сегодня: {notified_count}\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'Mode 🔄' or m.text.startswith('/mode'))
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    markup.add(
        types.KeyboardButton('Только НОВЫЕ 🔥'),
        types.KeyboardButton('ВСЕ активные 🟢'),
        types.KeyboardButton('🔙 Назад')
    )
    current = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
    bot.send_message(message.chat.id, f"🔄 <b>Выберите режим:</b>\nТекущий: {current}", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Только НОВЫЕ 🔥')
def set_mode_new(message):
    SETTINGS["SHOW_ONLY_NEW"] = True
    bot.reply_to(message, "✅ Режим: <b>Только НОВЫЕ</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'ВСЕ активные 🟢')
def set_mode_all(message):
    SETTINGS["SHOW_ONLY_NEW"] = False
    bot.reply_to(message, "✅ Режим: <b>ВСЕ активные</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

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

# Картинка
st.image("https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1000&auto=format&fit=crop", 
         use_container_width=True)

run_background_services()
st.success("✅ Сервер активен! Токен скрыт.")
st.write(f"Отправлено сигналов сегодня: {len(SETTINGS['NOTIFIED_TODAY'])}")
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300000, key="ref")
