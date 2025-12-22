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

# Импорты для работы с базой данных (Firestore)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    st.error("Пожалуйста, добавьте 'firebase-admin' в файл requirements.txt")

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ==========================================
# 1. НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ БД
# ==========================================

def init_firestore():
    try:
        # Извлекаем данные из системного окружения
        app_id = os.environ.get("__app_id", "default-app-id")
        fb_config_str = os.environ.get("__firebase_config", "{}")
        fb_config = json.loads(fb_config_str)
        project_id = fb_config.get("projectId")

        # Если project_id не найден в конфиге, попробуем использовать app_id или переменную окружения
        if not project_id:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

        try:
            firebase_admin.get_app()
        except ValueError:
            # Инициализация приложения
            if project_id:
                firebase_admin.initialize_app(options={'projectId': project_id})
            else:
                firebase_admin.initialize_app()
        
        # Явно передаем project_id в клиент firestore, чтобы избежать ошибки
        if project_id:
            db = firestore.client(project=project_id)
        else:
            db = firestore.client()
            
        # Путь согласно ПРАВИЛУ 1: /artifacts/{appId}/public/data/{collectionName}
        users_ref = db.collection('artifacts').document(app_id).collection('public').document('data').collection('users')
        return users_ref
    except Exception as e:
        # Если возникла ошибка, бот продолжит работу в оперативной памяти
        st.warning(f"База данных не подключена (используется память): {e}")
        return None

USERS_DB = init_firestore()

def load_approved_ids():
    ids = set()
    try:
        # Основной админ всегда имеет доступ
        admin_id = int(st.secrets.get("ADMIN_ID", 0))
        if admin_id != 0: ids.add(admin_id)
        
        if USERS_DB:
            # ПРАВИЛО 2: Простой запрос всех документов, фильтрация в коде
            docs = USERS_DB.stream()
            for doc in docs:
                data = doc.to_dict()
                if data.get('approved'):
                    ids.add(int(doc.id))
    except Exception as e:
        print(f"Error loading users: {e}")
    return ids

def save_user_to_cloud(user_id, approved=True):
    """Сохраняет или удаляет пользователя из Firebase"""
    if USERS_DB:
        try:
            doc_ref = USERS_DB.document(str(user_id))
            if approved:
                doc_ref.set({
                    'approved': True,
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
            else:
                # Если доступ отозван — полностью удаляем документ
                doc_ref.delete()
        except Exception as e:
            print(f"Cloud save error: {e}")

# --- ИНИЦИАЛИЗАЦИЯ БОТА ---

try:
    ADMIN_ID = int(st.secrets.get("ADMIN_ID", 0))
    TG_TOKEN = st.secrets["TG_TOKEN"]
except:
    ADMIN_ID = 0
    TG_TOKEN = os.environ.get("TG_TOKEN", "")

if not TG_TOKEN:
    st.error("❌ Токен не найден в Secrets!")
    st.stop()

bot = telebot.TeleBot(TG_TOKEN, threaded=True)

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
        "CHAT_IDS": set(), 
        "APPROVED_IDS": load_approved_ids(), # Загрузка из БД при старте
        "NOTIFIED_TODAY": set(),
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0,
        "TICKER_LIMIT": 500 
    }

SETTINGS = get_shared_state()

# ГЛОБАЛЬНОЕ СОСТОЯНИЕ ПРОГРЕССА
PROGRESS = {
    "current": 0, "total": 0, "running": False, "msg_id": None, "chat_id": None, "header": ""
}

def is_authorized(user_id):
    if ADMIN_ID != 0 and user_id == ADMIN_ID: return True
    return user_id in SETTINGS["APPROVED_IDS"]

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3, one_time_keyboard=False)
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
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        table = pd.read_html(io.StringIO(response.text))
        return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

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
        
        seq_states = [0]
        criticalLevel = df['Low'].iloc[0]
        cl = df['Close'].values
        for i in range(1, len(df)):
            prevS = seq_states[-1]
            if (prevS == 1 and cl[i] < criticalLevel): seq_states.append(-1)
            elif (prevS == -1 and cl[i] > criticalLevel): seq_states.append(1)
            else: seq_states.append(prevS)
        
        last = df.iloc[-1]; prev = df.iloc[-2]
        all_green_cur = (seq_states[-1] == 1) and (last['Close'] > last['SMA_Major']) and (last['ADX'] >= SETTINGS["ADX_THRESH"]) and (last['DI_Plus'] > last['DI_Minus'])
        all_green_prev = (seq_states[-2] == 1) and (prev['Close'] > prev['SMA_Major']) and (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and (prev['DI_Plus'] > prev['DI_Minus'])
        
        if all_green_cur and last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"]:
            if not SETTINGS["SHOW_ONLY_NEW"] or (all_green_cur and not all_green_prev):
                return {'ticker': ticker, 'price': last['Close'], 'atr': last['ATR_Pct'], 'is_new': (all_green_cur and not all_green_prev)}
    except: return None
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
                text = (f"{PROGRESS['header']}\nSMA: {SETTINGS['LENGTH_MAJOR']} | ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
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
            status_msg = bot.send_message(chat_id, "⏳ Инициализация...", parse_mode="HTML")
            PROGRESS.update({"current": 0, "total": total_tickers, "running": True, "msg_id": status_msg.message_id, "chat_id": chat_id, "header": "🚀 <b>Ручной поиск</b>"})
            threading.Thread(target=progress_updater, daemon=True).start()
        
        found_count = 0
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]: 
                PROGRESS["running"] = False
                break
            PROGRESS["current"] = i + 1
            res = check_ticker(t)
            if res:
                if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]: continue
                SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
                found_count += 1
                msg = f"{'🔥 NEW' if res['is_new'] else '🟢'} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
                targets = [chat_id] if is_manual else list(SETTINGS["CHAT_IDS"])
                for target in targets:
                    if is_authorized(target):
                        try: bot.send_message(target, msg, parse_mode="HTML")
                        except: pass
        
        PROGRESS["running"] = False
        final_text = f"✅ <b>Завершено</b>. Найдено: {found_count}" if found_count > 0 else f"🏁 <b>Завершено</b>. Ничего не найдено."
        if is_manual and chat_id:
            try: bot.edit_message_text(chat_id=chat_id, message_id=PROGRESS["msg_id"], text=final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            except: bot.send_message(chat_id, final_text, parse_mode="HTML", reply_markup=get_main_keyboard())
            
    except Exception as e: print(f"Scan error: {e}")
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

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_IDS"].add(message.chat.id)
    bot.send_message(message.chat.id, "👋 <b>Vova S&P 500 Screener</b>\nБот активен.", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['approve'])
def approve_user(message):
    if message.from_user.id != ADMIN_ID: return
    try:
        new_id = int(message.text.split()[1])
        SETTINGS["APPROVED_IDS"].add(new_id)
        save_user_to_cloud(new_id, True)
        bot.send_message(ADMIN_ID, f"✅ Пользователь {new_id} одобрен и сохранен в базе.")
        try: bot.send_message(new_id, "🎉 Администратор одобрил ваш доступ! Нажмите /start")
        except: pass
    except: bot.send_message(ADMIN_ID, "❌ Ошибка. Пример: /approve 12345678")

@bot.message_handler(commands=['revoke'])
def revoke_user(message):
    if message.from_user.id != ADMIN_ID: return
    try:
        old_id = int(message.text.split()[1])
        if old_id in SETTINGS["APPROVED_IDS"]:
            SETTINGS["APPROVED_IDS"].remove(old_id)
            save_user_to_cloud(old_id, False) # Удаляем из базы данных
            bot.send_message(ADMIN_ID, f"🚫 Доступ для {old_id} отозван и удален из базы.")
            try: bot.send_message(old_id, "⛔ Ваш доступ к боту был отозван администратором.")
            except: pass
        else:
            bot.send_message(ADMIN_ID, f"❓ Пользователь {old_id} не найден в списке одобренных.")
    except: bot.send_message(ADMIN_ID, "❌ Ошибка. Пример: /revoke 12345678")

@bot.message_handler(commands=['list_users'])
def list_users(message):
    if message.from_user.id != ADMIN_ID: return
    users = "\n".join([f"- <code>{u}</code>" for u in SETTINGS["APPROVED_IDS"]])
    bot.send_message(ADMIN_ID, f"👥 <b>Одобренные ID:</b>\n{users if users else 'Пусто'}", parse_mode="HTML")

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀')
def manual_scan(message):
    threading.Thread(target=perform_scan, args=(message.chat.id, True), daemon=True).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑')
def stop_scan(message):
    SETTINGS["STOP_SCAN"] = True
    bot.reply_to(message, "🛑 Останавливаю...")

@bot.message_handler(func=lambda m: m.text == 'Status 📊')
def get_status(message):
    mode = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все активные"
    bot.reply_to(message, f"⚙️ <b>Статус:</b>\nРежим: {mode}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML")

@bot.message_handler(func=lambda m: m.text == 'ATR 📉')
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('3.0 %', '5.0 %', '7.0 %', '10.0 %', '🔙 Назад')
    bot.send_message(message.chat.id, "📉 Выберите Max ATR:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'SMA 📈')
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('100', '150', '200', '🔙 Назад')
    bot.send_message(message.chat.id, "📈 Выберите SMA:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: '%' in m.text or m.text.isdigit())
def handle_values(message):
    if '%' in message.text:
        SETTINGS["MAX_ATR_PCT"] = float(message.text.replace(' %',''))
        bot.send_message(message.chat.id, f"✅ ATR: {SETTINGS['MAX_ATR_PCT']}%", reply_markup=get_main_keyboard())
    elif message.text.isdigit():
        SETTINGS["LENGTH_MAJOR"] = int(message.text)
        bot.send_message(message.chat.id, f"✅ SMA: {SETTINGS['LENGTH_MAJOR']}", reply_markup=get_main_keyboard())

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
st.success(f"✅ Бот активен. Одобрено пользователей в облаке: {len(SETTINGS['APPROVED_IDS'])}")
st.metric("Последний скан (Local)", SETTINGS["LAST_SCAN_TIME"])
