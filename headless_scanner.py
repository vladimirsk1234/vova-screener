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

# ВКЛЮЧАЕМ МНОГОПОТОЧНОСТЬ (threaded=True) - критически важно для работы кнопок во время скана
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
        # Уменьшаем объем загружаемых данных для ускорения
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 200: return None

        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        # Индикаторы
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

        # Sequence Logic (упрощенная для скорости)
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
        
        mode_txt = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
        header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"

        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        # 1. Отправляем СТАРТОВОЕ сообщение
        status_msg = None
        try:
            initial_text = (
                f"{header}\nРежим: {mode_txt}\n"
                f"SMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
                f"⏳ <b>Подготовка к сканированию {total_tickers} акций...</b>"
            )
            status_msg = bot.send_message(chat_id, initial_text, parse_mode="HTML", reply_markup=get_main_keyboard())
        except Exception as e:
            print(f"Failed to send start message: {e}")
        
        found_count = 0
        last_ui_update = time.time()
        
        # Основной цикл
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]:
                try: bot.send_message(chat_id, "🛑 Сканирование остановлено.")
                except: pass
                break
            
            # --- ОБНОВЛЕНИЕ ПРОГРЕССА ПО ТАЙМЕРУ (РАЗ В 5 СЕКУНД) ---
            # Это предотвращает Flood Limit от Telegram и не тормозит цикл
            now = time.time()
            if status_msg and (now - last_ui_update > 5):
                try:
                    progress_pct = int(((i + 1) / total_tickers) * 100)
                    bar_filled = int(progress_pct / 10)
                    bar_str = "▓" * bar_filled + "░" * (10 - bar_filled)
                    new_text = (
                        f"{header}\nРежим: {mode_txt}\n"
                        f"SMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
                        f"Всего: {total_tickers} акций\n\n"
                        f"⏳ Прогресс: {i+1}/{total_tickers} ({progress_pct}%)\n[{bar_str}]"
                    )
                    # Обновляем БЕЗ reply_markup для скорости
                    bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=new_text, parse_mode="HTML")
                    last_ui_update = now
                except Exception as e:
                    print(f"Progress update skipped: {e}")

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
        
        # --- ФИНАЛ ---
        final_text = f"✅ <b>Завершено</b>. Найдено: {found_count}" if found_count > 0 else f"🏁 <b>Завершено</b>. Ничего не найдено."
        if status_msg:
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=final_text, parse_mode="HTML")
            except:
                bot.send_message(chat_id, final_text, parse_mode="HTML")
        else:
            bot.send_message(chat_id, final_text, parse_mode="HTML")
            
    except Exception as e:
        print(f"Global scan error: {e}")
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
        "👋 <b>Vova S&P 500 Screener</b>\nБот активен. Нажмите кнопку меню ниже.",
        parse_mode="HTML", reply_markup=get_main_keyboard()
    )

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀' or m.text.startswith('/scan'))
def manual_scan(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    # Запускаем в отдельном потоке, чтобы не вешать бота
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

# Установка смещения времени
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

# Обработка выбора значений
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
# 4. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=30, long_polling_timeout=20)
        except: time.sleep(5)

def start_scheduler():
    while True:
        # Ждем час между авто-сканами
        time.sleep(3600)
        if SETTINGS["CHAT_ID"] and not SETTINGS["IS_SCANNING"]:
            perform_scan(SETTINGS["CHAT_ID"], False)

@st.cache_resource
def run_background_services():
    # Запуск бота
    t1 = threading.Thread(target=start_polling, daemon=True)
    t1.start()
    # Запуск планировщика
    t2 = threading.Thread(target=start_scheduler, daemon=True)
    t2.start()
    return True

# ==========================================
# 5. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.title("🤖 Vova Bot Server")

st.image("https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1000&auto=format&fit=crop", 
         use_container_width=True)

run_background_services()
st.success("✅ Сервер активен! Бот работает в фоновом режиме.")
st.write(f"Найдено сигналов за сегодня: {len(SETTINGS['NOTIFIED_TODAY'])}")
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60000, key="ref") # Рефреш страницы раз в минуту
