import logging
import asyncio
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, time
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from telegram.constants import ParseMode
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# === FIX FOR STREAMLIT ASYNCIO CONFLICT ===
import nest_asyncio
nest_asyncio.apply()
# ==========================================

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================

try:
    import streamlit as st
    try:
        if __name__ == '__main__':
            st.title("🤖 Vova Screener Bot is Running")
            if 'bot_logs' not in st.session_state:
                st.session_state.bot_logs = []
            
            # Простой вывод последних логов в UI для отладки
            st.write("### Логи последних событий:")
            for log in st.session_state.bot_logs[-5:]:
                st.text(log)
    except:
        pass

    TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
    ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
except (ImportError, FileNotFoundError, AttributeError):
    import os
    TG_TOKEN = os.environ.get("TG_TOKEN")
    ADMIN_ID = os.environ.get("ADMIN_ID")
    GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")

def log_ui(message):
    print(message) # В консоль
    if 'st' in globals() and 'bot_logs' in st.session_state:
        st.session_state.bot_logs.append(f"{datetime.now().strftime('%H:%M:%S')}: {message}")

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
    "scan_mode": "Top 10"
}

user_settings = {}
SENT_SIGNALS_CACHE = {"date": None, "tickers": set()}

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ТОЧНО КАК В WEB)
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

def calc_sma(series, length):
    return series.rolling(window=length).mean()

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_adx(df, length):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    
    # Избегаем деления на ноль
    tr_smooth = tr_smooth.replace(0, np.nan)
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    sum_di = plus_di + minus_di
    diff_di = (plus_di - minus_di).abs()
    dx = 100 * (diff_di / sum_di)
    
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calc_atr(df, length):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    return atr

# ==========================================
# 3. ЛОГИКА СТРАТЕГИИ (VOVA LOGIC)
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        # 1. Загрузка данных
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        # 2. Расчет индикаторов
        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_series, plus_di, minus_di = calc_adx(df, settings['adx_len'])
        atr_series = calc_atr(df, settings['atr_len'])
        
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        
        change = df['Close'].diff()
        efi_raw = change * df['Volume']
        df['EFI'] = calc_ema(efi_raw, settings['len_fast'])

        # 3. Инициализация переменных для цикла
        n = len(df)
        trend_state_list = [0] * n
        seq_state_list = [0] * n
        critical_level_list = [np.nan] * n
        peak_list = [np.nan] * n
        struct_ok_list = [False] * n

        seq_state = 0
        critical_level = np.nan
        seq_high = df['High'].iloc[0]
        seq_low = df['Low'].iloc[0]
        last_confirmed_peak = np.nan
        last_confirmed_trough = np.nan
        last_peak_was_hh = False 
        last_trough_was_hl = False

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
        
        # 4. Основной цикл по барам
        for i in range(1, n):
            c = close_arr[i]
            h = high_arr[i]
            l = low_arr[i]
            
            # --- Sequence Logic ---
            prev_seq_state = seq_state
            is_break = False
            
            if prev_seq_state == 1:
                if not np.isnan(critical_level):
                    is_break = c < critical_level 
            elif prev_seq_state == -1:
                if not np.isnan(critical_level):
                    is_break = c > critical_level 
            
            if is_break:
                if prev_seq_state == 1:
                    # Was UP, now DOWN
                    is_current_peak_hh = False
                    if not np.isnan(last_confirmed_peak):
                        if seq_high > last_confirmed_peak:
                            is_current_peak_hh = True
                    else:
                        is_current_peak_hh = True 
                    
                    last_peak_was_hh = is_current_peak_hh
                    last_confirmed_peak = seq_high
                    
                    seq_state = -1
                    seq_high = h
                    seq_low = l
                    critical_level = h
                else:
                    # Was DOWN, now UP
                    is_current_trough_hl = False
                    if not np.isnan(last_confirmed_trough):
                        if seq_low > last_confirmed_trough:
                            is_current_trough_hl = True
                    else:
                        is_current_trough_hl = True
                    
                    last_trough_was_hl = is_current_trough_hl
                    last_confirmed_trough = seq_low
                    
                    seq_state = 1
                    seq_high = h
                    seq_low = l
                    critical_level = l
            else:
                if seq_state == 1:
                    if h >= seq_high:
                        seq_high = h
                    if h >= seq_high: 
                         critical_level = l
                elif seq_state == -1:
                    if l <= seq_low:
                        seq_low = l
                    if l <= seq_low: 
                        critical_level = h
                else:
                    if c > seq_high:
                        seq_state = 1
                        critical_level = l
                    elif c < seq_low:
                        seq_state = -1
                        critical_level = h
                    else:
                        seq_high = max(seq_high, h)
                        seq_low = min(seq_low, l)

            # --- Super Trend Logic ---
            ema_imp_curr = ema_fast_vals[i]
            ema_imp_prev = ema_fast_vals[i-1]
            ema_slow_curr = ema_slow_vals[i]
            ema_slow_prev = ema_slow_vals[i-1]
            hist_curr = macd_hist_vals[i]
            hist_prev = macd_hist_vals[i-1]
            
            curr_adx = adx_vals[i]
            curr_pdi = pdi_vals[i]
            curr_mdi = mdi_vals[i]
            
            adx_strong = (curr_adx > settings['adx_thresh'])
            
            both_rising = (ema_imp_curr > ema_imp_prev) and (ema_slow_curr > ema_slow_prev)
            elder_bull = both_rising and (hist_curr > hist_prev)
            
            both_falling = (ema_imp_curr < ema_imp_prev) and (ema_slow_curr < ema_slow_prev)
            elder_bear = both_falling and (hist_curr < hist_prev)
            
            efi_bull = efi_vals[i] > 0
            efi_bear = efi_vals[i] < 0
            
            adx_bull = adx_strong and (curr_pdi > curr_mdi)
            adx_bear = adx_strong and (curr_mdi > curr_pdi)
            
            curr_trend_state = 0
            if adx_bull and elder_bull and efi_bull:
                curr_trend_state = 1
            elif adx_bear and elder_bear and efi_bear:
                curr_trend_state = -1
            
            trend_state_list[i] = curr_trend_state
            
            # --- Сохраняем состояние ---
            seq_state_list[i] = seq_state
            critical_level_list[i] = critical_level
            peak_list[i] = last_confirmed_peak
            struct_ok_list[i] = (last_peak_was_hh and last_trough_was_hl)

        # 5. Функция проверки условий (как в Web)
        def check_conditions(idx):
            if idx >= len(df) or idx < 0: return False, 0.0, np.nan, np.nan
            
            price = close_arr[idx]
            sma = df['SMA_Major'].iloc[idx]
            
            s_state = seq_state_list[idx]
            t_state = trend_state_list[idx]
            is_struct_ok = struct_ok_list[idx]
            
            crit = critical_level_list[idx]
            peak = peak_list[idx]
            
            c_seq = (s_state == 1)
            c_ma = (price > sma) if not np.isnan(sma) else False
            c_trend = (t_state != -1) # Не медвежий тренд (0 или 1) - в коде веба (t_state != -1)
            # Внимание: в веб-коде условие c_trend = (t_state != -1), то есть нейтральный или бычий.
            # Если нужно строго бычий, то (t_state == 1). Оставляю как в вебе.
            c_struct = is_struct_ok
            
            is_valid_setup = False
            rr_calc = 0.0
            
            if c_seq and c_ma and c_trend and c_struct:
                if not np.isnan(peak) and not np.isnan(crit):
                    risk = price - crit
                    reward = peak - price
                    
                    if risk > 0 and reward > 0:
                        rr_calc = reward / risk
                        is_valid_setup = True
            
            return is_valid_setup, rr_calc, crit, peak

        # 6. Финальная проверка (Сегодня и Вчера)
        is_valid_today, rr_today, sl_today, tp_today = check_conditions(n - 1)
        is_valid_yesterday, _, _, _ = check_conditions(n - 2)
        
        is_new = is_valid_today and (not is_valid_yesterday)
        
        # Фильтры
        if not is_valid_today: return None
        if rr_today < settings['min_rr']: return None
        
        # ATR check
        curr_c = close_arr[-1]
        curr_atr = atr_series.iloc[-1]
        atr_pct = (curr_atr / curr_c) * 100
        
        if atr_pct > settings['max_atr_pct']: return None
        
        # Расчет размера позиции
        risk = curr_c - sl_today
        risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
        shares = int(risk_amt / risk) if risk > 0 else 0
        max_sh = int(settings['portfolio_size'] / curr_c)
        shares = min(shares, max_sh)
        if shares < 1: shares = 1

        return {
            "Ticker": ticker, 
            "Price": curr_c, 
            "RR": rr_today, 
            "SL": sl_today, 
            "TP": tp_today,
            "ATR_SL": curr_c - curr_atr, 
            "Shares": shares, 
            "ATR_Pct": atr_pct, 
            "Is_New": is_new
        }
            
    except Exception as e:
        return None
    return None

def process_tickers(tickers, settings):
    results = []
    for ticker in tickers:
        res = run_strategy_for_ticker(ticker, settings)
        if res: results.append(res)
    results.sort(key=lambda x: x['RR'], reverse=True)
    return results

# ==========================================
# 4. БОТ: АВТОРИЗАЦИЯ И ХЕНДЛЕРЫ
# ==========================================

async def check_auth_async(user_id):
    """Асинхронная проверка прав доступа"""
    if ADMIN_ID and str(user_id) == str(ADMIN_ID):
        return True
    
    if not GITHUB_USERS_URL:
        # Если URL не настроен, пускаем только админа
        return False

    try:
        loop = asyncio.get_running_loop()
        # Выполняем requests в отдельном потоке, чтобы не блочить бота
        response = await loop.run_in_executor(None, requests.get, GITHUB_USERS_URL)
        if response.status_code == 200:
            allowed = [line.strip() for line in response.text.splitlines() if line.strip()]
            return str(user_id) in allowed
    except Exception as e:
        log_ui(f"Ошибка проверки доступа: {e}")
    return False

def get_settings(user_id):
    if user_id not in user_settings: user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return user_settings[user_id]

# --- Обработчики ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    log_ui(f"Получена команда /start от {user_id}")
    
    is_allowed = await check_auth_async(user_id)
    if not is_allowed:
        msg = (
            f"⛔ **Доступ запрещен!**\n\n"
            f"Ваш Telegram ID: `{user_id}`\n\n"
            f"Чтобы получить доступ, отправьте этот ID администратору: @Vova_Skl"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        return
    
    # Приветственное сообщение со всеми командами
    welcome_text = (
        "👋 **Добро пожаловать в Vova Screener Bot!**\n\n"
        "Я анализирую рынок и ищу лучшие точки входа (Structure Break + Trends).\n\n"
        "🛠 **Доступные команды:**\n\n"
        "🚀 **Запустить Скан**\n"
        "Мгновенный поиск сигналов в реальном времени.\n\n"
        "⚙️ **Настройки**\n"
        "Управление риск-менеджментом (Risk%, RR) и выбор рынка (Top 10 / S&P 500).\n\n"
        "🔄 **Авто-скан**\n"
        "Включите, чтобы я проверял рынок каждый час и присылал уведомления о новых сделках.\n\n"
        "👇 **Нажмите кнопку ниже, чтобы начать:**"
    )
    
    await show_main_menu(update, welcome_text)

async def any_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик любого текстового сообщения (если пользователь не нажал /start)"""
    if update.message:
        user_id = update.effective_user.id
        log_ui(f"Получено сообщение от {user_id}: {update.message.text}")
        # Перенаправляем на логику start
        await start_handler(update, context)

async def show_main_menu(update, text):
    keyboard = [
        [InlineKeyboardButton("🚀 Запустить Скан", callback_data="scan_now")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings_menu")],
        [InlineKeyboardButton("🔄 Авто-скан (Вкл/Выкл)", callback_data="toggle_auto")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    s = get_settings(user_id)
    text = (
        f"⚙️ **Настройки:**\n💰 Port: ${s['portfolio_size']} | ⚠️ Risk: {s['risk_per_trade_pct']}%\n"
        f"📊 RR: {s['min_rr']} | 🔍 Mode: {s['scan_mode']}\n"
        f"🔄 Auto: {'✅ ON' if s['auto_scan'] else '❌ OFF'}"
    )
    keyboard = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% 🔄", callback_data="change_risk")],
        [InlineKeyboardButton(f"RR: {s['min_rr']} 🔄", callback_data="change_rr")],
        [InlineKeyboardButton(f"Mode: {s['scan_mode']} 🔄", callback_data="change_mode")],
        [InlineKeyboardButton("🔙 Меню", callback_data="main_menu")]
    ]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    s = get_settings(user_id)
    data = query.data
    
    if data == "main_menu": await show_main_menu(update, "🤖 Главное меню")
    elif data == "settings_menu": await settings_menu(update, context)
    elif data == "scan_now": await run_scan_task(update, context, user_id, manual=True)
    elif data == "toggle_auto":
        s['auto_scan'] = not s['auto_scan']
        await query.message.reply_text(f"🔄 Авто-скан: {'ВКЛЮЧЕН' if s['auto_scan'] else 'ВЫКЛЮЧЕН'}")
        await show_main_menu(update, "🤖 Главное меню")
    elif data == "change_risk":
        s['risk_per_trade_pct'] = 0.5 if s['risk_per_trade_pct'] == 2.0 else s['risk_per_trade_pct'] + 0.5
        await settings_menu(update, context)
    elif data == "change_rr":
        s['min_rr'] = 1.0 if s['min_rr'] >= 3.0 else s['min_rr'] + 0.5
        await settings_menu(update, context)
    elif data == "change_mode":
        s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
        await settings_menu(update, context)

async def run_scan_task(update, context, user_id, manual=False):
    s = get_settings(user_id)
    msg_dest = update.callback_query.message if update.callback_query else context.bot
    if manual and update.callback_query:
        await update.callback_query.message.reply_text(f"🚀 Ищу новые сделки ({s['scan_mode']})...")

    tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, process_tickers, tickers, s)
    new_results = [r for r in results if r['Is_New']]
    
    if not new_results:
        if manual: await msg_dest.reply_text("Нет новых сделок за сегодня.")
        return

    if manual:
        for res in new_results: await send_signal_msg(context, user_id, res)

async def send_signal_msg(context, user_id, res):
    tv_link = f"https://www.tradingview.com/chart/?symbol={res['Ticker'].replace('-', '.')}"
    msg = (
        f"🔥 **[{res['Ticker']}]({tv_link})** | ${res['Price']:.2f}\n"
        f"🎯 **RR:** {res['RR']:.2f} | 🛑 **SL:** {res['SL']:.2f}\n"
        f"🏁 **TP:** {res['TP']:.2f} | 📦 **Lot:** {res['Shares']}"
    )
    await context.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    today_str = now.strftime("%Y-%m-%d")
    if SENT_SIGNALS_CACHE["date"] != today_str:
        SENT_SIGNALS_CACHE["date"] = today_str
        SENT_SIGNALS_CACHE["tickers"] = set()
    
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        for user_id, s in user_settings.items():
            if s.get('auto_scan', False):
                tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, process_tickers, tickers, s)
                candidates = [r for r in results if r['Is_New'] and r['Ticker'] not in SENT_SIGNALS_CACHE["tickers"]]
                if candidates:
                    await context.bot.send_message(chat_id=user_id, text="🔔 **Auto-Scan Signal**", parse_mode=ParseMode.MARKDOWN)
                    for res in candidates:
                        await send_signal_msg(context, user_id, res)
                        SENT_SIGNALS_CACHE["tickers"].add(res['Ticker'])

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
            # Добавляем ответ на ВСЕ сообщения (текст, фото и т.д.)
            application.add_handler(MessageHandler(filters.ALL, any_message_handler))
            
            job_queue = application.job_queue
            job_queue.run_repeating(auto_scan_job, interval=3600, first=10)
            
            log_ui("Бот запущен. Ожидание сообщений...")
            application.run_polling(stop_signals=[], drop_pending_updates=False)
        except Exception as e:
            log_ui(f"CRITICAL ERROR: {e}")
            if 'st' in globals(): st.error(f"Error: {e}")
    else:
        log_ui("Токен не найден.")
