import logging
import asyncio
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, time, timedelta
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from telegram.constants import ParseMode
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from streamlit_autorefresh import st_autorefresh # Для автообновления UI

# === FIX FOR STREAMLIT ASYNCIO CONFLICT ===
import nest_asyncio
nest_asyncio.apply()
# ==========================================

# ==========================================
# 0. ГЛОБАЛЬНОЕ СОСТОЯНИЕ (Для UI)
# ==========================================
# Используем глобальный словарь, чтобы данные были доступны
# и в потоке бота, и в интерфейсе Streamlit при перезагрузке
if 'BOT_STATE' not in globals():
    globals()['BOT_STATE'] = {
        "last_scan": None,
        "logs": []
    }
BOT_STATE = globals()['BOT_STATE']

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================

try:
    import streamlit as st
    try:
        if __name__ == '__main__':
            # --- UI STREAMLIT MONITORING ---
            # Авто-обновление страницы каждые 10 секунд
            st_autorefresh(interval=10000, key="monitor_refresh")

            st.title("🤖 Vova Screener Bot Monitor")
            
            # Получаем URL и токены (для отображения в UI используем secrets или env)
            tg_token_check = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
            gh_url_check = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
            
            # 1. Метрика: Пользователи
            col_u1, col_u2 = st.columns(2)
            if gh_url_check:
                try:
                    resp = requests.get(gh_url_check)
                    if resp.status_code == 200:
                        users_list = [l for l in resp.text.splitlines() if l.strip()]
                        col_u1.metric("✅ Авторизовано", f"{len(users_list)} юзеров")
                    else:
                        col_u1.error(f"GitHub Error: {resp.status_code}")
                except Exception as e:
                    col_u1.error("Ошибка сети")
            else:
                col_u1.warning("GitHub URL не задан")
            
            col_u2.metric("Статус Бота", "🟢 Работает" if tg_token_check else "🔴 Нет токена")

            # 2. Метрика: Время сканирования
            st.subheader("🕒 Статус Сканера")
            col_t1, col_t2 = st.columns(2)
            
            last_scan_time = BOT_STATE.get("last_scan")
            
            if last_scan_time:
                # Время последнего скана
                col_t1.metric("Последний авто-скан", last_scan_time.strftime("%H:%M:%S (NY)"))
                
                # Расчет времени до следующего (интервал 1 час)
                next_scan_time = last_scan_time + timedelta(hours=1)
                now_ny = datetime.now(pytz.timezone('US/Eastern'))
                
                # Разница
                delta = next_scan_time - now_ny
                total_seconds = delta.total_seconds()
                
                if total_seconds > 0:
                    mins = int(total_seconds // 60)
                    secs = int(total_seconds % 60)
                    col_t2.metric("До следующего скана", f"{mins} мин {secs} сек")
                else:
                    col_t2.metric("До следующего скана", "Запуск...")
            else:
                col_t1.metric("Последний авто-скан", "Ожидание...")
                col_t2.metric("До следующего скана", "Неизвестно")

            # 3. Логи
            st.subheader("📜 Последние логи")
            log_container = st.container(height=300)
            with log_container:
                # Показываем последние 20 логов в обратном порядке (новые сверху)
                for log in reversed(BOT_STATE["logs"][-20:]):
                    st.text(log)
            
            st.divider()
    except Exception as e:
        print(f"Streamlit UI Error: {e}")

    TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
    ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
except (ImportError, FileNotFoundError, AttributeError):
    import os
    TG_TOKEN = os.environ.get("TG_TOKEN")
    ADMIN_ID = os.environ.get("ADMIN_ID")
    GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")

def log_ui(message):
    print(message) # В консоль сервера
    # Добавляем в глобальный стейт с временной меткой
    ts = datetime.now().strftime('%H:%M:%S')
    BOT_STATE["logs"].append(f"[{ts}] {message}")
    # Чистим старые логи, чтобы не забивать память (храним 100)
    if len(BOT_STATE["logs"]) > 100:
        BOT_STATE["logs"] = BOT_STATE["logs"][-100:]

if not TG_TOKEN:
    log_ui("CRITICAL ERROR: TG_TOKEN not found!")
    if 'st' in globals():
        st.error("CRITICAL ERROR: TG_TOKEN not found! Check your secrets.")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Дефолтные настройки
DEFAULT_SETTINGS = {
    "portfolio_size": 10000,
    "risk_per_trade_pct": 1.0,
    "len_major": 200,
    "len_fast": 20,
    "len_slow": 40,
    "adx_len": 14,
    "adx_thresh": 20,
    "atr_len": 14,
    "min_rr": 1.5,
    "max_atr_pct": 5.0,
    "auto_scan": False,
    "scan_mode": "Top 10",
    "show_new_only": False 
}

user_settings = {}
# Множество ID пользователей, которые нажали "Стоп"
ABORT_SCAN_USERS = set()
# Состояния ввода для пользователей: {user_id: "WAITING_RISK" | "WAITING_RR" ...}
USER_STATES = {}
SENT_SIGNALS_CACHE = {"date": None, "tickers": set()}

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        html = pd.read_html(response.text, header=0)
        df = html[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

def get_top_10_tickers():
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO"]

def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ema(series, length): return series.ewm(span=length, adjust=False).mean()
def calc_atr(df, length):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_adx(df, length):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move, down_move = high - high.shift(1), low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    sum_di = plus_di + minus_di
    dx = 100 * ((plus_di - minus_di).abs() / sum_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di

# ==========================================
# 3. ЛОГИКА СТРАТЕГИИ
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_series, plus_di, minus_di = calc_adx(df, settings['adx_len'])
        atr_series = calc_atr(df, settings['atr_len'])
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        df['EFI'] = calc_ema(df['Close'].diff() * df['Volume'], settings['len_fast'])

        close_arr = df['Close'].values
        high_arr = df['High'].values
        low_arr = df['Low'].values
        ema_fast_vals = df['EMA_Fast'].values
        ema_slow_vals = df['EMA_Slow'].values
        macd_hist_vals = macd_hist.values
        efi_vals = df['EFI'].values
        adx_vals = adx_series.values
        pdi_vals = plus_di.values
        mdi_vals = minus_di.values

        n = len(df)
        trend_state_list = [0] * n
        seq_state_list = [0] * n
        critical_level_list = [np.nan] * n
        peak_list = [np.nan] * n
        struct_ok_list = [False] * n

        seq_state = 0
        critical_level = np.nan
        seq_high = high_arr[0]
        seq_low = low_arr[0]
        last_confirmed_peak = np.nan
        last_confirmed_trough = np.nan
        last_peak_was_hh = False 
        last_trough_was_hl = False

        for i in range(1, n):
            c, h, l = close_arr[i], high_arr[i], low_arr[i]
            prev_seq_state = seq_state
            is_break = False
            
            if prev_seq_state == 1:
                if not np.isnan(critical_level): is_break = c < critical_level 
            elif prev_seq_state == -1:
                if not np.isnan(critical_level): is_break = c > critical_level 
            
            if is_break:
                if prev_seq_state == 1:
                    is_current_peak_hh = True if np.isnan(last_confirmed_peak) else (seq_high > last_confirmed_peak)
                    last_peak_was_hh = is_current_peak_hh
                    last_confirmed_peak = seq_high
                    seq_state, seq_high, seq_low, critical_level = -1, h, l, h
                else:
                    is_current_trough_hl = True if np.isnan(last_confirmed_trough) else (seq_low > last_confirmed_trough)
                    last_trough_was_hl = is_current_trough_hl
                    last_confirmed_trough = seq_low
                    seq_state, seq_high, seq_low, critical_level = 1, h, l, l
            else:
                if seq_state == 1:
                    if h >= seq_high: seq_high = h
                    if h >= seq_high: critical_level = l
                elif seq_state == -1:
                    if l <= seq_low: seq_low = l
                    if l <= seq_low: critical_level = h
                else:
                    if c > seq_high: seq_state, critical_level = 1, l
                    elif c < seq_low: seq_state, critical_level = -1, h
                    else: seq_high, seq_low = max(seq_high, h), min(seq_low, l)

            adx_strong = (adx_vals[i] > settings['adx_thresh'])
            both_rising = (ema_fast_vals[i] > ema_fast_vals[i-1]) and (ema_slow_vals[i] > ema_slow_vals[i-1])
            elder_bull = both_rising and (macd_hist_vals[i] > macd_hist_vals[i-1])
            both_falling = (ema_fast_vals[i] < ema_fast_vals[i-1]) and (ema_slow_vals[i] < ema_slow_vals[i-1])
            elder_bear = both_falling and (macd_hist_vals[i] < macd_hist_vals[i-1])
            
            curr_trend_state = 0
            if adx_strong and (pdi_vals[i] > mdi_vals[i]) and elder_bull and (efi_vals[i] > 0):
                curr_trend_state = 1
            elif adx_strong and (mdi_vals[i] > pdi_vals[i]) and elder_bear and (efi_vals[i] < 0):
                curr_trend_state = -1
            
            trend_state_list[i] = curr_trend_state
            seq_state_list[i] = seq_state
            critical_level_list[i] = critical_level
            peak_list[i] = last_confirmed_peak
            struct_ok_list[i] = (last_peak_was_hh and last_trough_was_hl)

        def check_conditions(idx):
            if idx >= len(df) or idx < 0: return False, 0.0, np.nan, np.nan
            price, sma = close_arr[idx], df['SMA_Major'].iloc[idx]
            s_state, t_state = seq_state_list[idx], trend_state_list[idx]
            is_struct_ok = struct_ok_list[idx]
            crit, peak = critical_level_list[idx], peak_list[idx]
            
            c_seq = (s_state == 1)
            c_ma = (price > sma) if not np.isnan(sma) else False
            c_trend = (t_state != -1)
            
            is_valid_setup = False
            rr_calc = 0.0
            if c_seq and c_ma and c_trend and is_struct_ok:
                if not np.isnan(peak) and not np.isnan(crit):
                    risk, reward = price - crit, peak - price
                    if risk > 0 and reward > 0:
                        rr_calc = reward / risk
                        is_valid_setup = True
            return is_valid_setup, rr_calc, crit, peak

        is_valid_today, rr_today, sl_today, tp_today = check_conditions(n - 1)
        is_valid_yesterday, _, _, _ = check_conditions(n - 2)
        is_new = is_valid_today and (not is_valid_yesterday)
        
        if not is_valid_today: return None
        if rr_today < settings['min_rr']: return None
        
        curr_c = close_arr[-1]
        curr_atr = atr_series.iloc[-1]
        atr_pct = (curr_atr / curr_c) * 100
        
        if atr_pct > settings['max_atr_pct']: return None
        
        risk = curr_c - sl_today
        risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
        shares = int(risk_amt / risk) if risk > 0 else 0
        max_sh = int(settings['portfolio_size'] / curr_c)
        shares = min(shares, max_sh)
        if shares < 1: shares = 1

        return {
            "Ticker": ticker, "Price": curr_c, "RR": rr_today, "SL": sl_today, "TP": tp_today,
            "ATR_SL": curr_c - curr_atr, "Shares": shares, "ATR_Pct": atr_pct, "Is_New": is_new
        }
    except Exception as e:
        return None

# ==========================================
# 4. БОТ: АВТОРИЗАЦИЯ И ХЕНДЛЕРЫ
# ==========================================

async def check_auth_async(user_id):
    if ADMIN_ID and str(user_id) == str(ADMIN_ID): return True
    if not GITHUB_USERS_URL: return False
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, requests.get, GITHUB_USERS_URL)
        if response.status_code == 200:
            allowed = [line.strip() for line in response.text.splitlines() if line.strip()]
            return str(user_id) in allowed
    except: pass
    return False

def get_settings(user_id):
    if user_id not in user_settings: user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return user_settings[user_id]

# --- Клавиатуры ---
def get_main_keyboard(user_id):
    s = get_settings(user_id)
    auto_icon = "✅" if s['auto_scan'] else "❌"
    # Создаем постоянную клавиатуру внизу
    keyboard = [
        [KeyboardButton("🚀 Запустить Скан")],
        [KeyboardButton("⚙️ Настройки"), KeyboardButton(f"🔄 Авто: {auto_icon}")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# --- Обработчики ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await check_auth_async(user_id):
        await update.message.reply_text(f"⛔ Доступ запрещен. ID: `{user_id}`")
        return
    
    await update.message.reply_text(
        "👋 **Vova Screener Bot**\nМеню внизу 👇",
        reply_markup=get_main_keyboard(user_id),
        parse_mode=ParseMode.MARKDOWN
    )

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id
    
    if not await check_auth_async(user_id): return

    # Проверка на ввод значений (State Machine)
    if user_id in USER_STATES:
        state = USER_STATES[user_id]
        # Если пользователь нажал кнопку меню вместо ввода, отменяем ввод
        if text in ["🚀 Запустить Скан", "⚙️ Настройки"] or text.startswith("🔄 Авто:"):
            del USER_STATES[user_id]
            await update.message.reply_text("Ввод отменен.", reply_markup=get_main_keyboard(user_id))
            # Пропускаем дальше, чтобы обработать нажатие кнопки
        else:
            try:
                # Заменяем запятую на точку для удобства
                val = float(text.replace(',', '.'))
                s = get_settings(user_id)
                
                if state == "RISK": 
                    s['risk_per_trade_pct'] = val
                    await update.message.reply_text(f"✅ Risk обновлен: {val}%")
                elif state == "RR": 
                    s['min_rr'] = val
                    await update.message.reply_text(f"✅ Min RR обновлен: {val}")
                elif state == "PORT": 
                    s['portfolio_size'] = int(val)
                    await update.message.reply_text(f"✅ Portfolio обновлен: ${val}")
                elif state == "ATR":
                    s['max_atr_pct'] = val
                    await update.message.reply_text(f"✅ Max ATR обновлен: {val}%")
                
                del USER_STATES[user_id]
                await settings_menu(update, context) # Возвращаем меню настроек
                return
            except ValueError:
                await update.message.reply_text("❌ Пожалуйста, введите корректное число (например 1.5 или 10000).")
                return

    # Обработка команд меню
    if text == "🚀 Запустить Скан":
        await run_scan_task(update, context, user_id, manual=True)
    elif text == "⚙️ Настройки":
        await settings_menu(update, context)
    elif text.startswith("🔄 Авто:"):
        # Тоггл авто-скана через нижнее меню
        s = get_settings(user_id)
        s['auto_scan'] = not s['auto_scan']
        # Обновляем клавиатуру с новым статусом
        await update.message.reply_text(
            f"🔄 Авто-скан: {'ВКЛЮЧЕН' if s['auto_scan'] else 'ВЫКЛЮЧЕН'}", 
            reply_markup=get_main_keyboard(user_id)
        )
    else:
        # Для любых других сообщений показываем меню
        await update.message.reply_text("Выберите действие:", reply_markup=get_main_keyboard(user_id))

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Если вызов из callback (кнопки внутри сообщения)
    if update.callback_query:
        query = update.callback_query
        user_id = query.from_user.id
        msg_func = query.edit_message_text
    else:
        # Если вызов из текстовой команды
        user_id = update.effective_user.id
        msg_func = update.message.reply_text
        
    s = get_settings(user_id)
    text = (
        f"⚙️ **Настройки:**\n"
        f"💰 Portfolio: ${s['portfolio_size']} | ⚠️ Risk: {s['risk_per_trade_pct']}%\n"
        f"📊 RR: {s['min_rr']} | 🔍 Mode: {s['scan_mode']}\n"
        f"📈 Max ATR: {s['max_atr_pct']}%\n"
        f"👀 Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'}"
    )
    keyboard = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk")],
        [InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port")],
        [InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Mode: {s['scan_mode']} 🔄", callback_data="change_mode")],
        [InlineKeyboardButton(f"Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'} 🔄", callback_data="toggle_filter")],
    ]
    await msg_func(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    s = get_settings(user_id)
    data = query.data
    
    if data == "abort_scan":
        ABORT_SCAN_USERS.add(user_id)
        await query.message.reply_text("🛑 Сканирование остановлено пользователем.")
        return

    # Обработка запросов на ввод значений
    if data == "ask_risk":
        USER_STATES[user_id] = "RISK"
        await query.message.reply_text(f"Введите новый **Risk %** (текущий: {s['risk_per_trade_pct']}):", parse_mode=ParseMode.MARKDOWN)
        return
    elif data == "ask_rr":
        USER_STATES[user_id] = "RR"
        await query.message.reply_text(f"Введите новый **Min RR** (текущий: {s['min_rr']}):", parse_mode=ParseMode.MARKDOWN)
        return
    elif data == "ask_port":
        USER_STATES[user_id] = "PORT"
        await query.message.reply_text(f"Введите новый размер **Portfolio $** (текущий: {s['portfolio_size']}):", parse_mode=ParseMode.MARKDOWN)
        return
    elif data == "ask_atr":
        USER_STATES[user_id] = "ATR"
        await query.message.reply_text(f"Введите новый **Max ATR %** (текущий: {s['max_atr_pct']}):", parse_mode=ParseMode.MARKDOWN)
        return

    if data == "settings_menu": await settings_menu(update, context)
    elif data == "change_mode":
        s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
        await settings_menu(update, context)
    elif data == "toggle_filter":
        s['show_new_only'] = not s.get('show_new_only', False)
        await settings_menu(update, context)

async def run_scan_task(update, context, user_id, manual=False):
    s = get_settings(user_id)
    msg_dest = update.message if update.message else update.callback_query.message
    
    filter_mode = "🔥 Новые" if s.get('show_new_only', False) else "✅ Все активные"
    
    if user_id in ABORT_SCAN_USERS:
        ABORT_SCAN_USERS.remove(user_id)

    tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(tickers)
    
    # Сообщение с прогрессом и кнопкой Стоп
    progress_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="abort_scan")]])
    status_msg = await msg_dest.reply_text(
        f"🚀 Сканирование начато ({total} шт)...\nРежим: {filter_mode}",
        reply_markup=progress_keyboard
    )
    
    loop = asyncio.get_running_loop()
    found_count = 0
    
    # Разбиваем на батчи для обновления прогресса и возможности отмены
    batch_size = 5 
    
    for i in range(0, total, batch_size):
        # Проверка отмены
        if user_id in ABORT_SCAN_USERS:
            await status_msg.edit_text(f"🛑 Сканирование прервано на {i}/{total}.")
            ABORT_SCAN_USERS.remove(user_id)
            return

        batch_tickers = tickers[i : i + batch_size]
        
        # Запускаем батч в executor (чтобы не блокировать loop, но и не слишком нагружать)
        # Для простоты и стриминга делаем последовательно внутри батча
        for ticker in batch_tickers:
            if user_id in ABORT_SCAN_USERS: break # Двойная проверка
            
            # Расчет
            res = await loop.run_in_executor(None, run_strategy_for_ticker, ticker, s)
            
            if res:
                # Фильтрация
                is_pass = False
                if manual and not s.get('show_new_only', False): is_pass = True
                elif res['Is_New']: is_pass = True
                
                if is_pass:
                    found_count += 1
                    # ОТПРАВЛЯЕМ СРАЗУ
                    await send_signal_msg(context, user_id, res)
        
        # Обновление прогресса каждые batch_size тикеров
        pct = int((i + len(batch_tickers)) / total * 100)
        bar_len = 10
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        try:
            await status_msg.edit_text(
                f"🚀 Сканирование: {pct}%\n[{bar}] {i + len(batch_tickers)}/{total}\nНайдено: {found_count}",
                reply_markup=progress_keyboard
            )
        except: pass # Игнорируем ошибки редактирования (если не изменилось)

    final_text = f"✅ Сканирование завершено!\nВсего проверено: {total}\nНайдено сигналов: {found_count}"
    try:
        await status_msg.edit_text(final_text, reply_markup=None) # Убираем кнопку стоп
    except:
        await msg_dest.reply_text(final_text)

async def send_signal_msg(context, user_id, res):
    # TradingView format: . instead of - for BRK.B etc.
    tv_ticker = res['Ticker'].replace('-', '.')
    # Ссылка для открытия графика. На мобильных устройствах часто подхватывается приложением.
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
    
    status_icon = "🔥 NEW" if res['Is_New'] else "✅ ACTIVE"
    
    msg = (
        f"{status_icon} **[{tv_ticker}]({tv_link})** | ${res['Price']:.2f}\n"
        f"📊 **ATR:** {res['ATR_Pct']:.2f}% | **ATR SL:** ${res['ATR_SL']:.2f}\n"
        f"🎯 **RR:** {res['RR']:.2f} | 🛑 **SL:** ${res['SL']:.2f}\n"
        f"🏁 **TP:** ${res['TP']:.2f} | 📦 **Size:** {res['Shares']} stocks"
    )
    # disable_web_page_preview=True чтобы не загромождать превью графика, 
    # но можно и оставить False, если хотите видеть превью
    await context.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # --- ОБНОВЛЕНИЕ ГЛОБАЛЬНОГО СТАТУСА ---
    BOT_STATE["last_scan"] = now
    
    log_ui(f"🔄 Авто-скан: Запуск... {now.strftime('%H:%M:%S')}")

    today_str = now.strftime("%Y-%m-%d")
    if SENT_SIGNALS_CACHE["date"] != today_str:
        SENT_SIGNALS_CACHE["date"] = today_str
        SENT_SIGNALS_CACHE["tickers"] = set()
    
    # Авто-скан без прогресс-бара (фоновый)
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        for user_id, s in user_settings.items():
            if s.get('auto_scan', False):
                tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
                
                log_ui(f"🚀 Сканирую {len(tickers)} тикеров для {user_id}...")
                
                loop = asyncio.get_running_loop()
                # Давайте сделаем простой цикл для авто:
                for ticker in tickers:
                    res = await loop.run_in_executor(None, run_strategy_for_ticker, ticker, s)
                    if res and res['Is_New'] and res['Ticker'] not in SENT_SIGNALS_CACHE["tickers"]:
                         await send_signal_msg(context, user_id, res)
                         SENT_SIGNALS_CACHE["tickers"].add(res['Ticker'])
    else:
        log_ui(f"💤 Рынок закрыт (Time: {now.strftime('%H:%M')}). Пропуск.")

# ==========================================
# 5. SERVER & MAIN
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
    def log_message(self, format, *args): return

def start_keep_alive():
    try:
        server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
    except: pass

if __name__ == '__main__':
    start_keep_alive()
    if TG_TOKEN:
        try:
            log_ui("Инициализация бота...")
            application = ApplicationBuilder().token(TG_TOKEN).build()
            
            # Регистрируем хендлеры
            application.add_handler(CommandHandler('start', start_handler))
            application.add_handler(CallbackQueryHandler(button_handler))
            # Text Handler теперь обрабатывает и команды меню
            application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
            
            job_queue = application.job_queue
            job_queue.run_repeating(auto_scan_job, interval=3600, first=10)
            
            log_ui("Бот запущен. Ожидание сообщений...")
            application.run_polling(stop_signals=[], drop_pending_updates=False)
        except Exception as e:
            log_ui(f"CRITICAL ERROR: {e}")
            if 'st' in globals(): st.error(f"Error: {e}")
    else:
        log_ui("Токен не найден.")
