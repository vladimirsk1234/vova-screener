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
from streamlit_autorefresh import st_autorefresh

# === FIX FOR STREAMLIT ASYNCIO CONFLICT ===
import nest_asyncio
nest_asyncio.apply()
# ==========================================

# ==========================================
# 0. ГЛОБАЛЬНОЕ СОСТОЯНИЕ (Для UI)
# ==========================================
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
    
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {}
    if 'sent_signals_cache' not in st.session_state:
        st.session_state.sent_signals_cache = {"date": None, "tickers": set(), "last_auto_scan_ts": None}
    if 'user_states' not in st.session_state:
        st.session_state.user_states = {}
    if 'abort_scan_users' not in st.session_state:
        st.session_state.abort_scan_users = set()

    try:
        if __name__ == '__main__':
            st_autorefresh(interval=10000, key="monitor_refresh")
            st.title("🤖 Vova Screener Bot Monitor")
            
            tg_token_check = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
            gh_url_check = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
            
            col_u1, col_u2 = st.columns(2)
            if gh_url_check:
                try:
                    resp = requests.get(gh_url_check)
                    if resp.status_code == 200:
                        users_list = [l for l in resp.text.splitlines() if l.strip()]
                        col_u1.metric("✅ Авторизовано", f"{len(users_list)} юзеров")
                    else:
                        col_u1.error(f"GitHub Error: {resp.status_code}")
                except: col_u1.error("Ошибка сети")
            else:
                col_u1.warning("GitHub URL не задан")
            
            col_u2.metric("Статус Бота", "🟢 Работает" if tg_token_check else "🔴 Нет токена")

            st.subheader("🕒 Статус Сканера")
            col_t1, col_t2 = st.columns(2)
            last_scan_time = BOT_STATE.get("last_scan")
            
            # Логика отображения времени
            if last_scan_time:
                ny_time = last_scan_time.strftime("%H:%M:%S")
                col_t1.metric("Последний запуск (NY)", ny_time)
                
                # Таймер обратного отсчета (1 час)
                next_scan_time = last_scan_time + timedelta(hours=1)
                now_ny = datetime.now(pytz.timezone('US/Eastern'))
                # Приводим last_scan_time к timezone aware если он не такой
                if last_scan_time.tzinfo is None:
                     last_scan_time = pytz.timezone('US/Eastern').localize(last_scan_time)
                
                # Пересчитываем next_scan
                next_scan_time = last_scan_time + timedelta(hours=1)
                
                delta = next_scan_time - now_ny
                total_seconds = delta.total_seconds()
                
                if total_seconds > 0:
                    mins = int(total_seconds // 60)
                    secs = int(total_seconds % 60)
                    col_t2.metric("След. проверка через", f"{mins} мин {secs} сек")
                else:
                    col_t2.metric("След. проверка", "Выполняется...")
            else:
                col_t1.metric("Последний запуск", "Ожидание старта...")
                col_t2.metric("След. проверка", "Скоро...")

            st.subheader("📜 Последние логи")
            with st.container(height=300):
                for log in reversed(BOT_STATE["logs"][-20:]): st.text(log)
            st.divider()
    except Exception as e:
        print(f"UI Error: {e}")

    TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
    ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
except:
    import os
    TG_TOKEN = os.environ.get("TG_TOKEN")
    ADMIN_ID = os.environ.get("ADMIN_ID")
    GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")
    class MockSessionState(dict): pass
    if not hasattr(st, 'session_state'): st.session_state = MockSessionState()
    if 'user_settings' not in st.session_state: st.session_state.user_settings = {}
    if 'sent_signals_cache' not in st.session_state: st.session_state.sent_signals_cache = {"date": None, "tickers": set(), "last_auto_scan_ts": None}
    if 'user_states' not in st.session_state: st.session_state.user_states = {}
    if 'abort_scan_users' not in st.session_state: st.session_state.abort_scan_users = set()

def log_ui(message):
    print(message)
    ts = datetime.now().strftime('%H:%M:%S')
    BOT_STATE["logs"].append(f"[{ts}] {message}")
    if len(BOT_STATE["logs"]) > 100: BOT_STATE["logs"] = BOT_STATE["logs"][-100:]

if not TG_TOKEN:
    log_ui("CRITICAL ERROR: TG_TOKEN not found!")
    if 'st' in globals(): st.error("CRITICAL ERROR: TG_TOKEN not found!")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# === DEFAULT SETTINGS ===
DEFAULT_SETTINGS = {
    "portfolio_size": 100000,
    "risk_per_trade_pct": 0.5,
    "min_rr": 1.5,
    "len_major": 200,
    "len_fast": 20,
    "len_slow": 40,
    "adx_len": 14,
    "adx_thresh": 20,
    "atr_len": 14,
    "max_atr_pct": 5.0,
    "auto_scan": True,
    "scan_mode": "S&P 500",
    "show_new_only": True
}

user_settings = st.session_state.user_settings
ABORT_SCAN_USERS = st.session_state.abort_scan_users
USER_STATES = st.session_state.user_states
SENT_SIGNALS_CACHE = st.session_state.sent_signals_cache

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
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
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
    tr_smooth = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    sum_di = plus_di + minus_di
    diff_di = (plus_di - minus_di).abs()
    dx = 100 * (diff_di / sum_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di

def calc_atr(df, length):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    return atr

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
        change = df['Close'].diff()
        efi_raw = change * df['Volume']
        df['EFI'] = calc_ema(efi_raw, settings['len_fast'])

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
        trend_state_list, seq_state_list, critical_level_list, peak_list, struct_ok_list = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n
        seq_state, critical_level, seq_high, seq_low = 0, np.nan, df['High'].iloc[0], df['Low'].iloc[0]
        last_confirmed_peak, last_confirmed_trough, last_peak_was_hh, last_trough_was_hl = np.nan, np.nan, False, False

        for i in range(1, n):
            c, h, l = close_arr[i], high_arr[i], low_arr[i]
            prev_seq_state, is_break = seq_state, False
            if prev_seq_state == 1:
                if not np.isnan(critical_level): is_break = c < critical_level 
            elif prev_seq_state == -1:
                if not np.isnan(critical_level): is_break = c > critical_level 
            
            if is_break:
                if prev_seq_state == 1:
                    is_current_peak_hh = True if np.isnan(last_confirmed_peak) else (seq_high > last_confirmed_peak)
                    last_peak_was_hh, last_confirmed_peak = is_current_peak_hh, seq_high
                    seq_state, seq_high, seq_low, critical_level = -1, h, l, h
                else:
                    is_current_trough_hl = True if np.isnan(last_confirmed_trough) else (seq_low > last_confirmed_trough)
                    last_trough_was_hl, last_confirmed_trough = is_current_trough_hl, seq_low
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
            efi_bull, efi_bear = efi_vals[i] > 0, efi_vals[i] < 0
            adx_bull, adx_bear = adx_strong and (pdi_vals[i] > mdi_vals[i]), adx_strong and (mdi_vals[i] > pdi_vals[i])
            
            curr_trend_state = 0
            if adx_bull and elder_bull and efi_bull: curr_trend_state = 1
            elif adx_bear and elder_bear and efi_bear: curr_trend_state = -1
            
            trend_state_list[i], seq_state_list[i], critical_level_list[i], peak_list[i], struct_ok_list[i] = curr_trend_state, seq_state, critical_level, last_confirmed_peak, (last_peak_was_hh and last_trough_was_hl)

        def check_conditions(idx):
            if idx >= len(df) or idx < 0: return False, 0.0, np.nan, np.nan
            price, sma = close_arr[idx], df['SMA_Major'].iloc[idx]
            s_state, t_state = seq_state_list[idx], trend_state_list[idx]
            is_struct_ok = struct_ok_list[idx]
            crit, peak = critical_level_list[idx], peak_list[idx]
            c_seq, c_ma, c_trend = (s_state == 1), (price > sma) if not np.isnan(sma) else False, (t_state != -1)
            is_valid_setup, rr_calc = False, 0.0
            if c_seq and c_ma and c_trend and is_struct_ok:
                if not np.isnan(peak) and not np.isnan(crit):
                    risk, reward = price - crit, peak - price
                    if risk > 0 and reward > 0:
                        rr_calc, is_valid_setup = reward / risk, True
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
        
        risk_per_share = curr_c - sl_today
        shares = 0
        if risk_per_share > 0:
            risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
            shares = int(risk_amt / risk_per_share)
            max_sh = int(settings['portfolio_size'] / curr_c)
            shares = min(shares, max_sh)
            if shares < 1: shares = 1

        return {
            "Ticker": ticker, "Price": curr_c, "RR": rr_today, "SL": sl_today, "TP": tp_today,
            "ATR_SL": curr_c - curr_atr, "Shares": shares, "ATR_Pct": atr_pct, "Is_New": is_new
        }
    except Exception as e: return None

# ==========================================
# 4. БОТ: ЛОГИКА
# ==========================================

async def check_auth_async(user_id):
    if ADMIN_ID and str(user_id) == str(ADMIN_ID): return True
    if not GITHUB_USERS_URL: return False
    try:
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(None, requests.get, GITHUB_USERS_URL)
        return (str(user_id) in [l.strip() for l in r.text.splitlines() if l.strip()]) if r.status_code == 200 else False
    except: return False

def get_settings(user_id):
    if user_id not in user_settings: user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return user_settings[user_id]

def get_main_keyboard(user_id):
    s = get_settings(user_id)
    return ReplyKeyboardMarkup([
        [KeyboardButton("🚀 Запустить Скан")],
        [KeyboardButton("⚙️ Настройки"), KeyboardButton(f"🔄 Авто: {'✅' if s['auto_scan'] else '❌'}")],
        [KeyboardButton("ℹ️ Помощь")]
    ], resize_keyboard=True)

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await check_auth_async(uid):
        await update.message.reply_text(f"⛔ Доступ запрещен. ID: `{uid}`")
        return
    await update.message.reply_text("👋 **Vova Screener Bot**\nМеню внизу 👇", reply_markup=get_main_keyboard(uid), parse_mode=ParseMode.MARKDOWN)

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "ℹ️ **Справка по Vova Screener Bot**\n\n"
        "Бот ищет сигналы на покупку акций (S&P 500) по стратегии 'Vova Strategy' (Break of Structure + SuperTrend).\n\n"
        "🛠 **Параметры:**\n"
        "• **Portfolio**: Размер вашего депозита ($).\n"
        "• **Risk %**: Риск на одну сделку от депозита.\n"
        "• **RR (Risk/Reward)**: Минимальное соотношение Прибыль/Риск.\n"
        "• **ATR %**: Максимальная волатильность акции (чтобы избегать 'бешеных' бумаг).\n\n"
        "🔄 **Авто-скан:**\n"
        "Бот проверяет рынок **каждый час** (если рынок США открыт: 9:30-16:00 ET). Присылает только **новые** сигналы.\n\n"
        "🚀 **Ручной скан:**\n"
        "Запускает проверку прямо сейчас. Можно настроить фильтр 'Только новые' или 'Все активные'."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=get_main_keyboard(update.effective_user.id))

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id
    if not await check_auth_async(uid): return

    if uid in USER_STATES:
        state = USER_STATES[uid]
        if text in ["🚀 Запустить Скан", "⚙️ Настройки", "ℹ️ Помощь"] or text.startswith("🔄 Авто:"):
            del USER_STATES[uid]
            await update.message.reply_text("Ввод отменен.", reply_markup=get_main_keyboard(uid))
        else:
            try:
                clean_text = text.replace(',', '.').replace('%', '').replace('$', '').strip()
                val = float(clean_text)
                s = get_settings(uid)
                if state == "RISK": 
                    s['risk_per_trade_pct'] = val
                    await update.message.reply_text(f"✅ Risk обновлен: {val}%")
                elif state == "RR": 
                    s['min_rr'] = val
                    await update.message.reply_text(f"✅ Min RR обновлен: {val}")
                elif state == "PORT": 
                    s['portfolio_size'] = int(val)
                    await update.message.reply_text(f"✅ Portfolio обновлен: ${int(val)}")
                elif state == "ATR":
                    s['max_atr_pct'] = val
                    await update.message.reply_text(f"✅ Max ATR обновлен: {val}%")
                
                del USER_STATES[uid]
                await settings_menu(update, context) 
                return
            except ValueError:
                await update.message.reply_text("❌ Введите корректное число.")
                return

    if text == "🚀 Запустить Скан": await run_scan_process(context, uid, get_settings(uid), manual=True)
    elif text == "⚙️ Настройки": await settings_menu(update, context)
    elif text == "ℹ️ Помощь": await help_handler(update, context)
    elif text.startswith("🔄 Авто:"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await update.message.reply_text(f"🔄 Авто-скан: {'ВКЛЮЧЕН' if s['auto_scan'] else 'ВЫКЛЮЧЕН'}", reply_markup=get_main_keyboard(uid))
    else:
        try:
            float(text.replace(',', '.').replace('%', '').replace('$', '').strip())
            await update.message.reply_text("⚠️ Сначала нажмите кнопку в меню '⚙️ Настройки', чтобы изменить параметр.", reply_markup=get_main_keyboard(uid))
        except:
            await update.message.reply_text("Выберите действие:", reply_markup=get_main_keyboard(uid))

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if not update.callback_query else update.callback_query.from_user.id
    msg_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
    
    s = get_settings(uid)
    txt = (
        f"⚙️ **Настройки:**\n"
        f"💰 Portfolio: ${s['portfolio_size']:,} | ⚠️ Risk: {s['risk_per_trade_pct']}%\n"
        f"📊 RR: {s['min_rr']} | 🔍 Mode: {s['scan_mode']}\n"
        f"📈 Max ATR: {s['max_atr_pct']}%\n"
        f"👀 Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port")],
        [InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Mode: {s['scan_mode']} 🔄", callback_data="change_mode")],
        [InlineKeyboardButton(f"Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'} 🔄", callback_data="toggle_filter")]
    ]
    await msg_func(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    s = get_settings(uid)
    d = query.data
    
    if d == "abort_scan":
        ABORT_SCAN_USERS.add(uid)
        await query.message.reply_text("🛑 Сканирование остановлено.")
        return

    if d == "ask_risk":
        USER_STATES[uid] = "RISK"
        await query.message.reply_text(f"Введите **Risk %**:", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_rr":
        USER_STATES[uid] = "RR"
        await query.message.reply_text(f"Введите **Min RR**:", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_port":
        USER_STATES[uid] = "PORT"
        await query.message.reply_text(f"Введите **Portfolio $**:", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_atr":
        USER_STATES[uid] = "ATR"
        await query.message.reply_text(f"Введите **Max ATR %**:", parse_mode=ParseMode.MARKDOWN)
    elif d == "change_mode":
        s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
        await settings_menu(update, context)
    elif d == "toggle_filter":
        s['show_new_only'] = not s.get('show_new_only', False)
        await settings_menu(update, context)

# --- UNIFIED SCAN FUNCTION ---
async def run_scan_process(context, uid, s, manual=False, is_auto=False):
    if is_auto:
        last_ts = SENT_SIGNALS_CACHE.get("last_auto_scan_ts")
        if last_ts and (datetime.now() - last_ts).total_seconds() < 1800: return 
        SENT_SIGNALS_CACHE["last_auto_scan_ts"] = datetime.now()

    if uid in ABORT_SCAN_USERS: ABORT_SCAN_USERS.remove(uid)
    
    tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(tickers)
    
    filter_txt = "🔥 Новые" if s.get('show_new_only', False) else "✅ Все"
    title = f"🔄 Авто-скан: {filter_txt}" if is_auto else f"🚀 Скан: {filter_txt}"
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="abort_scan")]])
    status_msg = await context.bot.send_message(chat_id=uid, text=f"{title}\nОжидание...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    batch = 5
    
    for i in range(0, total, batch):
        if uid in ABORT_SCAN_USERS:
            await status_msg.edit_text(f"🛑 Прервано на {i}/{total}.")
            ABORT_SCAN_USERS.remove(uid)
            return
            
        for t in tickers[i:i+batch]:
            if uid in ABORT_SCAN_USERS: break
            res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
            if res:
                is_pass = False
                if is_auto:
                    if res['Is_New']: is_pass = True
                else:
                    if not s.get('show_new_only', False): is_pass = True
                    elif res['Is_New']: is_pass = True
                
                if is_auto and is_pass:
                    if res['Ticker'] in SENT_SIGNALS_CACHE["tickers"]: is_pass = False
                    else: SENT_SIGNALS_CACHE["tickers"].add(res['Ticker'])

                if is_pass:
                    found += 1
                    await send_signal_msg(context, uid, res)
        
        pct = int((i+len(tickers[i:i+batch]))/total*100)
        filled = int(10 * pct / 100)
        bar = "█"*filled + "░"*(10-filled)
        try: await status_msg.edit_text(f"{title}\nПрогресс: {pct}%\n[{bar}] {i+len(tickers[i:i+batch])}/{total}\nНайдено: {found}", reply_markup=pkb)
        except: pass

    final_txt = f"✅ {title} завершен!\nНайдено сигналов: {found}"
    try: 
        await status_msg.edit_text(final_txt, reply_markup=None)
        # Show menu after finish
        await context.bot.send_message(chat_id=uid, text="Готово! Возвращаюсь в меню...", reply_markup=get_main_keyboard(uid))
    except: pass

async def send_signal_msg(context, uid, res):
    tv_t = res['Ticker'].replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_t}"
    icon = "🔥 NEW" if res['Is_New'] else "✅ ACTIVE"
    
    msg = (
        f"{icon} **[{tv_t}]({tv_link})** | ${res['Price']:.2f}\n"
        f"📊 **ATR:** {res['ATR_Pct']:.2f}% | **ATR SL:** ${res['ATR_SL']:.2f}\n"
        f"🎯 **RR:** {res['RR']:.2f} | 🛑 **SL:** ${res['SL']:.2f}\n"
        f"🏁 **TP:** ${res['TP']:.2f} | 📦 **Size:** {res['Shares']} stocks"
    )
    await context.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Store aware datetime
    if now.tzinfo is None:
        now = pytz.timezone('US/Eastern').localize(now)
    BOT_STATE["last_scan"] = now
    
    today = now.strftime("%Y-%m-%d")
    if SENT_SIGNALS_CACHE["date"] != today:
        SENT_SIGNALS_CACHE["date"] = today
        SENT_SIGNALS_CACHE["tickers"] = set()
    
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        log_ui(f"🔄 Auto-Scan Start... {now.strftime('%H:%M')}")
        for uid, s in user_settings.items():
            if s.get('auto_scan', False):
                await run_scan_process(context, uid, s, manual=False, is_auto=True)
    else:
        log_ui(f"💤 Market Closed {now.strftime('%H:%M')}")

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
    def log_message(self, format, *args): return

def start_keep_alive():
    try:
        s = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
        threading.Thread(target=s.serve_forever, daemon=True).start()
    except: pass

if __name__ == '__main__':
    start_keep_alive()
    if TG_TOKEN:
        try:
            log_ui("Bot Init...")
            app = ApplicationBuilder().token(TG_TOKEN).build()
            app.add_handler(CommandHandler('start', start_handler))
            app.add_handler(CallbackQueryHandler(button_handler))
            app.add_handler(CommandHandler('help', help_handler)) # Added command
            app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
            app.job_queue.run_repeating(auto_scan_job, interval=3600, first=10)
            log_ui("Polling Started...")
            app.run_polling(stop_signals=[], drop_pending_updates=False)
        except Exception as e:
            log_ui(f"ERR: {e}")
    else: log_ui("No Token")
