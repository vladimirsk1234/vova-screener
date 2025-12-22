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

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ==========================================
# 1. НАСТРОЙКИ И ЗАГРУЗКА ПОЛЬЗОВАТЕЛЕЙ
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

# Создаем сессию для yfinance, чтобы избежать блокировок (Anti-bot)
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

bot = telebot.TeleBot(TG_TOKEN, threaded=True)

@st.cache_resource
def get_shared_state():
    return {
        "LENGTH_MAJOR": 200, 
        "MAX_ATR_PCT": 7.0, # Ослаблено (было 5.0)
        "ADX_THRESH": 15,   # Ослаблено (было 20)
        "MIN_RR": 1.2,      # Ослаблено (было 1.5)
        "AUTO_SCAN_INTERVAL": 3600, 
        "IS_SCANNING": False, 
        "STOP_SCAN": False,
        "SHOW_ONLY_NEW": False, # По умолчанию показываем всё для теста
        "LAST_SCAN_TIME": "Никогда",
        "CHAT_IDS": set(), "APPROVED_IDS": fetch_approved_ids(), 
        "NOTIFIED_TODAY": set(), "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0, "TICKER_LIMIT": 500 
    }

SETTINGS = get_shared_state()
PROGRESS = {"current": 0, "total": 0, "running": False, "msg_id": None, "chat_id": None, "header": ""}

def is_authorized(user_id):
    if ADMIN_ID != 0 and user_id == ADMIN_ID: return True
    return user_id in SETTINGS["APPROVED_IDS"]

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3, one_time_keyboard=False)
    markup.row(types.KeyboardButton('Scan 🚀'), types.KeyboardButton('Stop 🛑'))
    markup.row(types.KeyboardButton('Status 📊'), types.KeyboardButton('Mode 🔄'))
    markup.row(types.KeyboardButton('ATR 📉'), types.KeyboardButton('SMA 📈'), types.KeyboardButton('RR ⚖️'))
    markup.row(types.KeyboardButton('Time 🕒'))
    return markup

def get_local_now():
    return datetime.utcnow() + timedelta(hours=SETTINGS["TIMEZONE_OFFSET"])

# ==========================================
# 2. ФУНКЦИИ АНАЛИЗА
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        table = pd.read_html(io.StringIO(response.text))
        tickers = [str(t).replace('.', '-').strip() for t in table[0]['Symbol'].tolist()]
        return sorted(list(set(tickers)))
    except Exception as e:
        print(f"Scraper error: {e}")
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "ARES", "GOOGL", "AMZN", "META"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker, verbose=False):
    try:
        df = None
        for attempt in range(3):
            df = yf.download(ticker, period="2y", interval="1d", progress=False, 
                             auto_adjust=True, timeout=15, session=YF_SESSION)
            if not df.empty and len(df) >= 250: break
            time.sleep(0.5)

        if df is None or df.empty or len(df) < 250:
            return None

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

        # 1. SMA Major
        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        # 2. ATR (RMA ДЛЯ ТОЧНОСТИ)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR_Val'] = pine_rma(df['TR'], 14) 
        df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
        
        # 3. ADX & DI
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        tr = pine_rma(df['TR'], 14)
        p_dm = pine_rma(df['+DM'], 14); m_dm = pine_rma(df['-DM'], 14)
        df['DI_Plus'] = 100 * (p_dm / tr); df['DI_Minus'] = 100 * (m_dm / tr)
        dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = pine_rma(dx, 14)
        
        # 4. SEQUENCE LOGIC (ОГРАНИЧЕНИЕ ОКНА ДО 300 БАРОВ)
        df_seq = df.tail(300).copy()
        seq_states = []
        seqState = 0; seqHigh = df_seq['High'].iloc[0]; seqLow = df_seq['Low'].iloc[0]; criticalLevel = df_seq['Low'].iloc[0]
        cl_vals = df_seq['Close'].values; hi_vals = df_seq['High'].values; lo_vals = df_seq['Low'].values
        
        for i in range(len(df_seq)):
            if i == 0:
                seq_states.append(0); continue
            c, h, l = cl_vals[i], hi_vals[i], lo_vals[i]
            prevS = seq_states[-1]
            isBreak = (prevS == 1 and c < criticalLevel) or (prevS == -1 and c > criticalLevel)
            if isBreak:
                if prevS == 1: seqState = -1; seqHigh = h; seqLow = l; criticalLevel = h
                else: seqState = 1; seqHigh = h; seqLow = l; criticalLevel = l
            else:
                seqState = prevS
                if seqState == 1:
                    if h >= seqHigh: seqHigh = h; criticalLevel = l
                elif seqState == -1:
                    if l <= seqLow: seqLow = l; criticalLevel = h
                else:
                    if c > seqHigh: seqState = 1; criticalLevel = l
                    elif c < seqLow: seqState = -1; criticalLevel = h
                    else: seqHigh = math.max(seqHigh, h); seqLow = min(seqLow, l)
            seq_states.append(seqState)
        
        last = df.iloc[-1]; prev = df.iloc[-2]
        if pd.isna(last['ADX']): return None
        
        # Условия стратегии
        cond_seq = (seq_states[-1] == 1)
        cond_ma = (last['Close'] > last['SMA_Major'])
        cond_trend = (last['ADX'] >= SETTINGS["ADX_THRESH"]) and (last['DI_Plus'] > last['DI_Minus'])
        all_green_cur = cond_seq and cond_ma and cond_trend
        
        # Проверка новизны
        all_green_prev = (seq_states[-2] == 1) and (prev['Close'] > prev['SMA_Major']) and (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and (prev['DI_Plus'] > prev['DI_Minus'])
        is_new_signal = all_green_cur and not all_green_prev
        
        # 5. КОРРЕКТНЫЙ RISK REWARD (С РАСШИРЕННОЙ ЦЕЛЬЮ)
        current_price = float(last['Close'])
        
        # Используем предложенную логику расширенного таргета, если HH слишком близко
        target_price = max(seqHigh, current_price + (current_price - criticalLevel) * SETTINGS["MIN_RR"])
        
        if criticalLevel >= current_price: # Стоп выше цены - это ошибка/не лонг
            rr_ratio = 0
        else:
            risk = current_price - criticalLevel
            reward = target_price - current_price
            rr_ratio = round(reward / risk, 2)
        
        # Фильтры
        pass_atr = (last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"])
        pass_rr = (rr_ratio >= SETTINGS["MIN_RR"])

        result_data = {
            'ticker': ticker, 'price': current_price, 'atr': last['ATR_Pct'], 
            'is_new': is_new_signal, 'rr': rr_ratio, 'tp': target_price, 'sl': criticalLevel,
            'adx': round(last['ADX'], 2), 'sma': round(last['SMA_Major'], 2),
            'lights': { 'seq': cond_seq, 'ma': cond_ma, 'trend': cond_trend },
            'all_green': all_green_cur, 'pass_atr': pass_atr, 'pass_rr': pass_rr
        }

        if verbose:
            # Добавляем отчет о причинах провала для режима диагностики
            result_data["fail_reasons"] = {
                "sequence": cond_seq,
                "sma_200": cond_ma,
                "adx_trend": cond_trend,
                "atr_limit": pass_atr,
                "rr_ratio": pass_rr
            }
            return result_data

        if all_green_cur and pass_atr and pass_rr:
            return result_data
                
    except Exception as e:
        if verbose: print(f"Error {ticker}: {e}")
        return None
    return None

# ==========================================
# 3. УПРАВЛЕНИЕ СКАНЕРОМ
# ==========================================

def progress_updater():
    while PROGRESS["running"]:
        try:
            if PROGRESS["total"] > 0:
                pct = int((PROGRESS["current"] / PROGRESS["total"]) * 100)
                bar_str = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
                text = (f"{PROGRESS['header']}\nSMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}% | R:R: 1:{SETTINGS['MIN_RR']}\n"
                        f"Прогресс: {PROGRESS['current']}/{PROGRESS['total']} ({pct}%)\n[{bar_str}]")
                bot.edit_message_text(chat_id=PROGRESS["chat_id"], message_id=PROGRESS["msg_id"], text=text, parse_mode="HTML")
        except: pass
        time.sleep(5)

def perform_scan(chat_id=None, is_manual=False):
    if is_manual and chat_id and not is_authorized(chat_id): return
    if SETTINGS["IS_SCANNING"]:
        if chat_id: bot.send_message(chat_id, "⚠️ Сканирование уже идет!")
        return
    
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    try:
        local_now = get_local_now()
        current_date_str = local_now.strftime("%Y-%m-%d")
        if SETTINGS["LAST_DATE"] != current_date_str:
            SETTINGS["NOTIFIED_TODAY"] = set()
            SETTINGS["LAST_DATE"] = current_date_str
        
        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        if is_manual and chat_id:
            status_msg = bot.send_message(chat_id, f"⏳ Поиск среди {total_tickers} акций S&P 500...", parse_mode="HTML")
            PROGRESS.update({"current": 0, "total": total_tickers, "running": True, "msg_id": status_msg.message_id, "chat_id": chat_id, "header": "🚀 <b>Ручной поиск</b>"})
            threading.Thread(target=progress_updater, daemon=True).start()
        
        found_count = 0
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]: 
                PROGRESS["running"] = False
                break
            
            PROGRESS["current"] = i + 1
            if i > 0 and i % 15 == 0: time.sleep(1.0) # Анти-флуд
            
            res = check_ticker(t)
            if res:
                # Фильтрация новизны только для АВТО-скана
                if not is_manual and SETTINGS["SHOW_ONLY_NEW"] and not res['is_new']:
                    continue
                
                # Пропускаем уже отправленные сегодня (только для авто-скана)
                if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]: 
                    continue
                
                if not is_manual: SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
                found_count += 1
                
                msg = (f"{'🔥 NEW' if res['is_new'] else '🟢'} <b>{res['ticker']}</b> | ${res['price']:.2f}\n"
                       f"📊 ATR: {res['atr']:.2f}% | <b>R:R: 1:{res['rr']}</b>\n"
                       f"🎯 TP: ${res['tp']:.2f} | 🛑 SL: ${res['sl']:.2f}")
                
                targets = [chat_id] if is_manual else list(SETTINGS["CHAT_IDS"])
                for target in targets:
                    if is_authorized(target):
                        try: bot.send_message(target, msg, parse_mode="HTML")
                        except: pass
        
        PROGRESS["running"] = False
        final_text = f"✅ <b>Поиск завершен</b>. Найдено: {found_count}" if found_count > 0 else f"🏁 <b>Завершено</b>. Активных сигналов не найдено."
        if is_manual and chat_id:
            try: bot.edit_message_text(chat_id=chat_id, message_id=PROGRESS["msg_id"], text=final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            except: bot.send_message(chat_id, final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            
    except Exception as e: print(f"Global Scan error: {e}")
    finally:
        PROGRESS["running"] = False
        SETTINGS["IS_SCANNING"] = False
        SETTINGS["LAST_SCAN_TIME"] = get_local_now().strftime("%H:%M:%S")

# ==========================================
# 4. ОБРАБОТЧИКИ ТЕЛЕГРАМ
# ==========================================

@bot.message_handler(func=lambda m: not is_authorized(m.from_user.id))
def unauthorized_access(message):
    bot.send_message(message.chat.id, 
        f"⛔ <b>Доступ ограничен.</b>\n\nВаш ID: <code>{message.from_user.id}</code>\n"
        f"Отправьте этот ID администратору <b>@Vova_Skl</b> для получения доступа.", parse_mode="HTML")

@bot.message_handler(commands=['check'])
def diagnostic_check(message):
    if not is_authorized(message.from_user.id): return
    try:
        parts = message.text.split()
        ticker = parts[1].upper().strip() if len(parts) > 1 else ""
        if not ticker:
            bot.reply_to(message, "❌ Укажите тикер. Пример: `/check ARES`", parse_mode="Markdown")
            return
        
        bot.send_message(message.chat.id, f"🔍 Диагностика <b>{ticker}</b>...", parse_mode="HTML")
        info = check_ticker(ticker, verbose=True)
        if not info:
            bot.send_message(message.chat.id, "❌ Ошибка загрузки данных Yahoo.")
            return

        l = info['lights']
        fr = info['fail_reasons']
        
        report = (
            f"📊 <b>Отчет по {ticker}:</b>\n"
            f"Цена: ${info['price']:.2f} (SMA{SETTINGS['LENGTH_MAJOR']}: {info['sma']})\n\n"
            f"{'🟢' if l['ma'] else '🔴'} Price &gt; SMA: {info['price'] > info['sma']}\n"
            f"{'🟢' if l['seq'] else '🔴'} Sequence state: {'BULL' if l['seq'] else 'BEAR/NEUTRAL'}\n"
            f"{'🟢' if l['trend'] else '🔴'} Trend (ADX {info['adx']} &gt; {SETTINGS['ADX_THRESH']}): {l['trend']}\n\n"
            f"<b>Фильтры:</b>\n"
            f"{'✅' if info['pass_atr'] else '❌'} ATR ({info['atr']:.2f}%) &lt;= {SETTINGS['MAX_ATR_PCT']}%\n"
            f"{'✅' if info['pass_rr'] else '❌'} R:R (1:{info['rr']}) &gt;= 1:{SETTINGS['MIN_RR']}\n\n"
            f"🎯 TP: ${info['tp']:.2f} | 🛑 SL: ${info['sl']:.2f}\n"
            f"🆕 Новый сигнал: {'ДА' if info['is_new'] else 'НЕТ'}\n\n"
            f"⚠️ <b>Вердикт:</b> {'ПРОХОДИТ ✅' if info['all_green'] and info['pass_atr'] and info['pass_rr'] else 'ОТКЛОНЕН ❌'}"
        )
        bot.send_message(message.chat.id, report, parse_mode="HTML")
    except Exception as e: bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_IDS"].add(message.chat.id)
    bot.send_message(message.chat.id, "👋 <b>Vova S&P 500 Screener Pro</b>\nБот активен.", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['reload'])
def reload_users(message):
    if message.from_user.id != ADMIN_ID: return
    SETTINGS["APPROVED_IDS"] = fetch_approved_ids()
    bot.send_message(ADMIN_ID, f"✅ Список обновлен из GitHub ({len(SETTINGS['APPROVED_IDS'])} чел.)")

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀')
def manual_scan_btn(message):
    threading.Thread(target=perform_scan, args=(message.chat.id, True), daemon=True).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑')
def stop_scan(message):
    SETTINGS["STOP_SCAN"] = True
    bot.reply_to(message, "🛑 Останавливаю...")

@bot.message_handler(func=lambda m: m.text == 'Status 📊')
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все активные"
    bot.reply_to(message, f"⚙️ <b>Статус:</b>\nРежим: {mode}\nОдобрено: {len(SETTINGS['APPROVED_IDS'])}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nMin R:R: 1:{SETTINGS['MIN_RR']}\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML")

@bot.message_handler(func=lambda m: m.text == 'ATR 📉')
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('5.0 %', '7.0 %', '8.0 %', '10.0 %', '🔙 Назад')
    bot.send_message(message.chat.id, "📉 Выберите Max ATR:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'SMA 📈')
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('100', '150', '200', '🔙 Назад')
    bot.send_message(message.chat.id, "📈 Выберите SMA:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'RR ⚖️')
def open_rr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('RR 1:1.2', 'RR 1:1.5', 'RR 1:2.0', '🔙 Назад')
    bot.send_message(message.chat.id, "⚖️ Минимальный R:R:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Mode 🔄')
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('Только НОВЫЕ 🔥', 'ВСЕ активные 🟢', '🔙 Назад')
    bot.send_message(message.chat.id, "🔄 Режим поиска:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: '%' in m.text or m.text.isdigit() or 'НОВЫЕ' in m.text or 'ВСЕ' in m.text or 'RR 1:' in m.text)
def handle_values(message):
    if '%' in message.text:
        SETTINGS["MAX_ATR_PCT"] = float(message.text.replace(' %',''))
        bot.send_message(message.chat.id, f"✅ ATR установлен: {SETTINGS['MAX_ATR_PCT']}%", reply_markup=get_main_keyboard())
    elif message.text.isdigit():
        SETTINGS["LENGTH_MAJOR"] = int(message.text)
        bot.send_message(message.chat.id, f"✅ SMA установлен: {SETTINGS['LENGTH_MAJOR']}", reply_markup=get_main_keyboard())
    elif 'RR 1:' in message.text:
        SETTINGS["MIN_RR"] = float(message.text.replace('RR 1:',''))
        bot.send_message(message.chat.id, f"✅ R:R: 1:{SETTINGS['MIN_RR']}", reply_markup=get_main_keyboard())
    elif 'НОВЫЕ' in message.text:
        SETTINGS["SHOW_ONLY_NEW"] = True
        bot.send_message(message.chat.id, "✅ Режим: Только НОВЫЕ", reply_markup=get_main_keyboard())
    elif 'ВСЕ' in message.text:
        SETTINGS["SHOW_ONLY_NEW"] = False
        bot.send_message(message.chat.id, "✅ Режим: ВСЕ активные", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'Time 🕒')
def check_time(message):
    local_time = get_local_now().strftime("%H:%M")
    bot.reply_to(message, f"🕒 Ваше время: <b>{local_time}</b> (UTC{SETTINGS['TIMEZONE_OFFSET']})", parse_mode="HTML")

# ==========================================
# 5. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=30)
        except: time.sleep(5)

def start_scheduler():
    while True:
        time.sleep(3600)
        if SETTINGS["CHAT_IDS"] and not SETTINGS["IS_SCANNING"]: perform_scan(is_manual=False)

@st.cache_resource
def run_background_services():
    threading.Thread(target=start_polling, daemon=True).start()
    threading.Thread(target=start_scheduler, daemon=True).start()
    return True

st.title("🤖 Vova Bot Server")
run_background_services()
st.success(f"✅ Бот активен. Пользователей: {len(SETTINGS['APPROVED_IDS'])}")
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])
