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

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================

# Попытка загрузить секреты из Streamlit secrets или переменных окружения
try:
    import streamlit as st
    # Если запущен как Streamlit App, показываем статус
    try:
        if __name__ == '__main__':
            st.title("🤖 Vova Screener Bot is Running")
            st.write("Бот работает в фоновом режиме.")
    except:
        pass

    TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
    ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
except (ImportError, FileNotFoundError, AttributeError):
    # Fallback для локального запуска без Streamlit
    import os
    TG_TOKEN = os.environ.get("TG_TOKEN")
    ADMIN_ID = os.environ.get("ADMIN_ID")
    GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")

if not TG_TOKEN:
    print("CRITICAL ERROR: TG_TOKEN not found in secrets or environment variables.")

# Настройки логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Дефолтные настройки пользователя
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
    "auto_scan": False, # Включено ли авто-сканирование
    "scan_mode": "Top 10" # 'Top 10' или 'S&P 500'
}

# Хранилище настроек в памяти
user_settings = {}

# Анти-спам кэш: { "date": "YYYY-MM-DD", "tickers": set() }
SENT_SIGNALS_CACHE = {
    "date": None,
    "tickers": set()
}

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СТРАТЕГИИ
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
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"] # Fallback

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
    
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di

def calc_atr(df, length):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()

# ==========================================
# 3. ОСНОВНАЯ ЛОГИКА СТРАТЕГИИ
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        # Загружаем данные
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']:
            return None

        # Индикаторы
        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_series, plus_di, minus_di = calc_adx(df, settings['adx_len'])
        atr_series = calc_atr(df, settings['atr_len'])
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        df['EFI'] = calc_ema(df['Close'].diff() * df['Volume'], settings['len_fast'])

        # Расчет логики (упрощенный цикл для скорости)
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        sma = df['SMA_Major'].values
        adx = adx_series.values
        pdi = plus_di.values
        mdi = minus_di.values
        hist = macd_hist.values
        efi = df['EFI'].values
        ema_f = df['EMA_Fast'].values
        ema_s = df['EMA_Slow'].values
        
        n = len(df)
        
        seq_state = 0
        crit_level = np.nan
        seq_high = high[0]
        seq_low = low[0]
        peak = np.nan
        trough = np.nan
        hh = False
        hl = False
        
        valid_history = [] 
        
        for i in range(1, n):
            is_break = False
            c, h, l = close[i], high[i], low[i]
            
            if seq_state == 1 and not np.isnan(crit_level) and c < crit_level: is_break = True
            elif seq_state == -1 and not np.isnan(crit_level) and c > crit_level: is_break = True
            
            if is_break:
                if seq_state == 1: 
                    is_cur_hh = seq_high > peak if not np.isnan(peak) else True
                    hh, peak = is_cur_hh, seq_high
                    seq_state, seq_high, seq_low, crit_level = -1, h, l, h
                else: 
                    is_cur_hl = seq_low > trough if not np.isnan(trough) else True
                    hl, trough = is_cur_hl, seq_low
                    seq_state, seq_high, seq_low, crit_level = 1, h, l, l
            else:
                if seq_state == 1:
                    if h >= seq_high: seq_high, crit_level = h, l
                elif seq_state == -1:
                    if l <= seq_low: seq_low, crit_level = h
                else:
                    if c > seq_high: seq_state, crit_level = 1, l
                    elif c < seq_low: seq_state, crit_level = -1, h
                    else: seq_high, seq_low = max(seq_high, h), min(seq_low, l)

            adx_ok = (adx[i] > settings['adx_thresh'])
            trend_bull = adx_ok and (pdi[i] > mdi[i]) and \
                         (ema_f[i] > ema_f[i-1]) and (ema_s[i] > ema_s[i-1]) and \
                         (hist[i] > hist[i-1]) and (efi[i] > 0)
            
            trend_ok = trend_bull 
            ma_ok = (c > sma[i]) if not np.isnan(sma[i]) else False
            struct_ok = hh and hl
            
            is_valid = False
            rr = 0.0
            
            if seq_state == 1 and ma_ok and trend_ok and struct_ok and not np.isnan(peak):
                risk = c - crit_level
                reward = peak - c
                if risk > 0 and reward > 0:
                    rr = reward / risk
                    is_valid = True
            
            valid_history.append(is_valid)

        today_valid = valid_history[-1]
        yesterday_valid = valid_history[-2] if len(valid_history) > 1 else False
        is_new = today_valid and not yesterday_valid
        
        if today_valid:
            curr_c = close[-1]
            curr_atr = atr_series.iloc[-1]
            atr_pct = (curr_atr / curr_c) * 100
            
            if atr_pct > settings['max_atr_pct']: return None
            
            sl_struct = crit_level
            tp = peak
            risk = curr_c - sl_struct
            reward = tp - curr_c
            rr = reward / risk
            
            if rr < settings['min_rr']: return None

            risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
            shares = 0
            if risk > 0:
                shares = int(risk_amt / risk)
                max_sh = int(settings['portfolio_size'] / curr_c)
                shares = min(shares, max_sh)
                if shares < 1: shares = 1
            
            atr_sl_rec = curr_c - curr_atr
            
            return {
                "Ticker": ticker,
                "Price": curr_c,
                "RR": rr,
                "SL": sl_struct,
                "TP": tp,
                "ATR_SL": atr_sl_rec,
                "Shares": shares,
                "ATR_Pct": atr_pct,
                "Is_New": is_new
            }
            
    except Exception as e:
        return None
    return None

# ==========================================
# 4. ФУНКЦИИ БОТА (HANDLERS)
# ==========================================

def check_auth(user_id):
    if str(user_id) == str(ADMIN_ID):
        return True
    try:
        response = requests.get(GITHUB_USERS_URL)
        if response.status_code == 200:
            allowed = [line.strip() for line in response.text.splitlines() if line.strip()]
            return str(user_id) in allowed
        else:
            logger.error(f"Ошибка GitHub: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ошибка авторизации: {e}")
        return False

def get_settings(user_id):
    if user_id not in user_settings:
        user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return user_settings[user_id]

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_auth(user_id):
        await update.message.reply_text(f"⛔ Доступ запрещен. Ваш ID: `{user_id}`.", parse_mode=ParseMode.MARKDOWN)
        return
    await show_main_menu(update, "🤖 **Screener Vova Bot**\nМеню управления:")

async def show_main_menu(update, text):
    keyboard = [
        [InlineKeyboardButton("🚀 Запустить Скан (Manual)", callback_data="scan_now")],
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
    
    if data == "main_menu":
        await show_main_menu(update, "🤖 Главное меню")
    elif data == "settings_menu":
        await settings_menu(update, context)
    elif data == "scan_now":
        await run_scan_task(update, context, user_id, manual=True)
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
        await update.callback_query.message.reply_text(f"🚀 Ищу ВСЕ новые сделки за сегодня ({s['scan_mode']})...")

    tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, process_tickers, tickers, s)
    
    # Фильтруем только Is_New (валидные сегодня, которых не было вчера)
    # В ручном режиме показываем ВСЕ new, в авто - только те, что не слали сегодня
    new_results = [r for r in results if r['Is_New']]
    
    if not new_results:
        if manual: await msg_dest.reply_text("Нет новых сделок за сегодня.")
        return

    # В ручном режиме (Manual) показываем ВСЕ найденные
    # В авто режиме это обрабатывается в auto_scan_job
    if manual:
        for res in new_results:
            await send_signal_msg(context, user_id, res)

def process_tickers(tickers, settings):
    results = []
    for ticker in tickers:
        res = run_strategy_for_ticker(ticker, settings)
        if res: results.append(res)
    results.sort(key=lambda x: x['RR'], reverse=True)
    return results

async def send_signal_msg(context, user_id, res):
    tv_link = f"https://www.tradingview.com/chart/?symbol={res['Ticker'].replace('-', '.')}"
    msg = (
        f"🔥 **[{res['Ticker']}]({tv_link})** | ${res['Price']:.2f}\n"
        f"🎯 **RR:** {res['RR']:.2f}\n"
        f"🛑 **SL:** {res['SL']:.2f} | 🏁 **TP:** {res['TP']:.2f}\n"
        f"📦 **Позиция:** {res['Shares']} шт\n"
        f"💡 **ATR SL Rec:** {res['ATR_SL']:.2f}"
    )
    await context.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)

# --- Auto Scan Background Job ---
async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    today_str = now.strftime("%Y-%m-%d")
    
    # Сброс кэша отправленных, если наступил новый день
    if SENT_SIGNALS_CACHE["date"] != today_str:
        SENT_SIGNALS_CACHE["date"] = today_str
        SENT_SIGNALS_CACHE["tickers"] = set()
    
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    if now.weekday() < 5 and market_open <= now.time() <= market_close:
        for user_id, s in user_settings.items():
            if s.get('auto_scan', False):
                tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, process_tickers, tickers, s)
                
                # Фильтр: Только новые (Is_New) И которых нет в кэше отправленных сегодня
                candidates = [r for r in results if r['Is_New'] and r['Ticker'] not in SENT_SIGNALS_CACHE["tickers"]]
                
                if candidates:
                    await context.bot.send_message(chat_id=user_id, text="🔔 **Авто-скан: Новые сигналы**", parse_mode=ParseMode.MARKDOWN)
                    for res in candidates:
                        await send_signal_msg(context, user_id, res)
                        SENT_SIGNALS_CACHE["tickers"].add(res['Ticker']) # Запоминаем

# ==========================================
# 5. KEEP-ALIVE SERVER
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Bot is alive')
    # Отключаем логирование запросов, чтобы не мусорить в консоли
    def log_message(self, format, *args): return

def start_keep_alive():
    try:
        server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        print("Keep-alive server started on port 8080")
    except Exception as e:
        print(f"Failed to start keep-alive server: {e}")

# ==========================================
# 6. ЗАПУСК
# ==========================================
if __name__ == '__main__':
    start_keep_alive() # Запуск сервера для "Live" статуса
    
    if TG_TOKEN:
        application = ApplicationBuilder().token(TG_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CallbackQueryHandler(button_handler))
        
        job_queue = application.job_queue
        job_queue.run_repeating(auto_scan_job, interval=3600, first=10) # Каждый час
        
        print("Бот запущен...")
        application.run_polling()
    else:
        print("Бот НЕ запущен: Отсутствует TG_TOKEN.")
