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
import nest_asyncio

# Применяем патч для asyncio
nest_asyncio.apply()

# ==========================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ (SINGLETON)
# ==========================================
import streamlit as st

class BotGlobalState:
    def __init__(self):
        self.last_scan = None
        self.logs = []
        self.user_settings = {}
        self.sent_signals_cache = {"date": None, "tickers": set(), "last_auto_scan_ts": None}
        self.user_states = {}
        self.abort_scan_users = set()
        self.bot_thread = None

    def add_log(self, message):
        print(message)
        ts = datetime.now().strftime('%H:%M:%S')
        self.logs.append(f"[{ts}] {message}")
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

@st.cache_resource
def get_global_state():
    return BotGlobalState()

STATE = get_global_state()

# ==========================================
# 2. НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ==========================================
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

def get_settings(user_id):
    if user_id not in STATE.user_settings:
        STATE.user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return STATE.user_settings[user_id]

# ==========================================
# 3. INTERFACE (STREAMLIT MONITOR)
# ==========================================
try:
    if __name__ == '__main__':
        st_autorefresh(interval=10000, key="monitor_refresh")
        st.title("🤖 Vova Screener Bot Monitor")
        
        try:
            TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
            GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
            ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
        except:
            TG_TOKEN = os.environ.get("TG_TOKEN")
            GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")
            ADMIN_ID = os.environ.get("ADMIN_ID")

        c1, c2 = st.columns(2)
        if GITHUB_USERS_URL:
            try:
                r = requests.get(GITHUB_USERS_URL)
                if r.status_code == 200:
                    ul = [l for l in r.text.splitlines() if l.strip()]
                    c1.metric("✅ Users Approved", f"{len(ul)}")
                else: c1.error(f"GH Error: {r.status_code}")
            except: c1.error("Net Error")
        else: c1.warning("No GH URL")
            
        bs = "🟢 Active" if (STATE.bot_thread and STATE.bot_thread.is_alive()) else "🔴 Stopped"
        c2.metric("Bot Service", bs)

        st.subheader("🕒 Status and Timers")
        ct1, ct2 = st.columns(2)
        if STATE.last_scan:
            ny = pytz.timezone('US/Eastern')
            ls = STATE.last_scan if STATE.last_scan.tzinfo else ny.localize(STATE.last_scan)
            ls_ny = ls.astimezone(ny)
            ct1.metric("Last Auto Scan (NY)", ls_ny.strftime("%H:%M:%S"))
            
            nxt = ls_ny + timedelta(hours=1)
            now_ny = datetime.now(ny)
            rem = nxt - now_ny
            if rem.total_seconds() > 0:
                m, s = divmod(int(rem.total_seconds()), 60)
                ct2.metric("Time to Next Scan", f"{m}m {s}s")
            else:
                ct2.metric("Time to Next Scan", "Pending...")
        else:
            ct1.metric("Last Auto Scan", "No data")
            ct2.metric("Time to Next Scan", "Awaiting start")

        st.subheader("📜 Live Logs")
        with st.container(height=300):
            for l in reversed(STATE.logs[-20:]): st.text(l)
        st.divider()
except: pass

# ==========================================
# 4. HELPERS (EXACT FROM WEB SCREENER)
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        h = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=h)
        df = pd.read_html(r.text, header=0)[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

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
# 5. STRATEGY (100% EXACT LOGIC FROM WEB SCREENER)
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        # Metadata for bot
        pe = t_obj.info.get('trailingPE', 'N/A')
        if pe != 'N/A': pe = f"{pe:.2f}"

        # --- Предварительные расчеты (Web Screener Logic) ---
        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_series, plus_di, minus_di = calc_adx(df, settings['adx_len'])
        atr_series = calc_atr(df, settings['atr_len'])
        
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        
        change = df['Close'].diff()
        efi_raw = change * df['Volume']
        df['EFI'] = calc_ema(efi_raw, settings['len_fast'])

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

        for i in range(1, n):
            c = close_arr[i]
            h = high_arr[i]
            l = low_arr[i]
            
            # --- Sequence Logic (Web Screener Logic) ---
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
                    is_current_peak_hh = False
                    if not np.isnan(last_confirmed_peak):
                        if seq_high > last_confirmed_peak:
                            is_current_peak_hh = True
                    else:
                        is_current_peak_hh = True 
                    
                    last_peak_was_hh = is_current_peak_hh
                    last_confirmed_peak = seq_high
                    seq_state, seq_high, seq_low, critical_level = -1, h, l, h
                else:
                    is_current_trough_hl = False
                    if not np.isnan(last_confirmed_trough):
                        if seq_low > last_confirmed_trough:
                            is_current_trough_hl = True
                    else:
                        is_current_trough_hl = True
                    
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

            # --- Super Trend Logic (Web Screener Logic) ---
            ema_imp_curr, ema_imp_prev = ema_fast_vals[i], ema_fast_vals[i-1]
            ema_slow_curr, ema_slow_prev = ema_slow_vals[i], ema_slow_vals[i-1]
            hist_curr, hist_prev = macd_hist_vals[i], macd_hist_vals[i-1]
            curr_adx, curr_pdi, curr_mdi = adx_vals[i], pdi_vals[i], mdi_vals[i]
            
            adx_strong = (curr_adx > settings['adx_thresh'])
            both_rising = (ema_imp_curr > ema_imp_prev) and (ema_slow_curr > ema_slow_prev)
            elder_bull = both_rising and (hist_curr > hist_prev)
            both_falling = (ema_imp_curr < ema_imp_prev) and (ema_slow_curr < ema_slow_prev)
            elder_bear = both_falling and (hist_curr < hist_prev)
            
            efi_bull, efi_bear = efi_vals[i] > 0, efi_vals[i] < 0
            adx_bull, adx_bear = adx_strong and (curr_pdi > curr_mdi), adx_strong and (curr_mdi > curr_pdi)
            
            curr_trend_state = 0
            if adx_bull and elder_bull and efi_bull: curr_trend_state = 1
            elif adx_bear and elder_bear and efi_bear: curr_trend_state = -1
            
            trend_state_list[i] = curr_trend_state
            seq_state_list[i] = seq_state
            critical_level_list[i] = critical_level
            peak_list[i] = last_confirmed_peak
            struct_ok_list[i] = (last_peak_was_hh and last_trough_was_hl)

        # --- Internal Check Logic ---
        def check_conditions(idx):
            if idx >= len(df) or idx < 0: return False, 0.0, np.nan, np.nan
            price, sma = close_arr[idx], df['SMA_Major'].iloc[idx]
            s_state, t_state = seq_state_list[idx], trend_state_list[idx]
            is_struct_ok = struct_ok_list[idx]
            crit, peak = critical_level_list[idx], peak_list[idx]
            
            c_seq = (s_state == 1)
            c_ma = (price > sma) if not np.isnan(sma) else False
            c_trend = (t_state != -1) # Not Bearish (Neutral or Bullish)
            c_struct = is_struct_ok
            
            is_valid_setup, rr_calc = False, 0.0
            if c_seq and c_ma and c_trend and c_struct:
                if not np.isnan(peak) and not np.isnan(crit):
                    risk, reward = price - crit, peak - price
                    if risk > 0 and reward > 0:
                        rr_calc, is_valid_setup = reward / risk, True
            return is_valid_setup, rr_calc, crit, peak

        # --- Final Validation ---
        is_valid_today, rr_today, sl_today, tp_today = check_conditions(n - 1)
        is_valid_yesterday, _, _, _ = check_conditions(n - 2)
        is_new = is_valid_today and (not is_valid_yesterday)
        
        if not is_valid_today or rr_today < settings['min_rr']: return None
        
        curr_c = close_arr[-1]
        curr_atr = atr_series.iloc[-1]
        atr_pct = (curr_atr / curr_c) * 100
        if atr_pct > settings['max_atr_pct']: return None
        
        # Trade Size
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
            "ATR_SL": curr_c - curr_atr, "Shares": shares, "ATR_Pct": atr_pct, 
            "Is_New": is_new, "PE": pe
        }
    except Exception: return None

# ==========================================
# 6. BOT HANDLERS
# ==========================================
async def check_auth_async(uid):
    if ADMIN_ID and str(uid) == str(ADMIN_ID): return True
    if not GITHUB_USERS_URL: return False
    try:
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(None, requests.get, GITHUB_USERS_URL)
        return (str(uid) in [l.strip() for l in r.text.splitlines() if l.strip()]) if r.status_code == 200 else False
    except: return False

def get_main_kb(uid):
    s = get_settings(uid)
    return ReplyKeyboardMarkup([
        [KeyboardButton("🚀 Запустить Скан")],
        [KeyboardButton("⚙️ Настройки"), KeyboardButton(f"🔄 Авто: {'✅' if s['auto_scan'] else '❌'}")],
        [KeyboardButton("ℹ️ Помощь")]
    ], resize_keyboard=True)

async def start_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id
    if not await check_auth_async(uid):
        await u.message.reply_text(f"⛔ Access Denied. ID: `{uid}`\nSend this ID to @Vova_Skl to get access.")
        return
    welcome = (
        "👋 **Vova Screener Bot (Exact Web Logic)**\n\n"
        "🛠 **Controls:**\n"
        "1. Click **Scan** to check current US market.\n"
        "2. Use **Settings** to adjust risk/portfolio.\n"
        "3. Turn on **Auto-Scan** for notifications."
    )
    await u.message.reply_text(welcome, reply_markup=get_main_kb(uid), parse_mode=ParseMode.MARKDOWN)

async def help_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = (
        "ℹ️ **Help Menu**\n\n"
        "**Strategy:** Break of Structure + SuperTrend (SMA 200 Filter).\n\n"
        "**Parameters:**\n"
        "• **Portfolio Size**: Your total capital ($).\n"
        "• **Risk %**: Risk per trade from total capital.\n"
        "• **RR**: Target Reward vs Risk ratio.\n"
        "• **ATR %**: Volatility limit filter.\n\n"
        "🔄 **Auto-Scan**: Runs hourly during NYSE open hours."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Configuration:**\n\n"
        f"💰 **Portfolio**: ${s['portfolio_size']:,}\n"
        f"⚠️ **Risk Per Trade**: {s['risk_per_trade_pct']}%\n"
        f"📊 **Minimum RR**: {s['min_rr']}\n"
        f"🔍 **Market Mode**: {s['scan_mode']}\n"
        f"👀 **Filter**: {'🔥 Only New' if s['show_new_only'] else '✅ Show All'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port"),
         InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Market: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥 New' if s['show_new_only'] else '✅ All'} 🔄", callback_data="ch_filt")],
        [InlineKeyboardButton("ℹ️ HELP INFO", callback_data="show_help")]
    ]
    await func(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def btn_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    uid = q.from_user.id
    s = get_settings(uid)
    d = q.data
    
    if d == "stop":
        STATE.abort_scan_users.add(uid)
        await q.message.reply_text("🛑 Stopping scanning process...")
        return

    if d == "show_help": await help_h(u, c); return

    if d.startswith("ask_"):
        STATE.user_states[uid] = d.split("_")[1].upper()
        await q.message.reply_text(f"Type the new value for **{STATE.user_states[uid]}**:", parse_mode=ParseMode.MARKDOWN)
        return

    if d == "ch_mode": s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
    elif d == "ch_filt": s['show_new_only'] = not s['show_new_only']
    
    await settings_menu(u, c)

async def txt_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = u.message.text
    uid = u.effective_user.id
    if not await check_auth_async(uid): return

    if uid in STATE.user_states:
        st_code = STATE.user_states[uid]
        if txt.startswith(("🚀", "⚙️", "ℹ️", "🔄")):
            del STATE.user_states[uid]
            await u.message.reply_text("Cancelled.", reply_markup=get_main_kb(uid))
        else:
            try:
                val = float(txt.replace(',', '.').replace('%', '').replace('$', '').strip())
                s = get_settings(uid)
                if st_code == "RISK": s['risk_per_trade_pct'] = val
                elif st_code == "RR": s['min_rr'] = val
                elif st_code == "PORT": s['portfolio_size'] = int(val)
                elif st_code == "ATR": s['max_atr_pct'] = val
                del STATE.user_states[uid]
                await u.message.reply_text(f"✅ Saved: {val}")
                await settings_menu(u, c)
                return
            except:
                await u.message.reply_text("❌ Input error.")
                return

    if txt == "🚀 Запустить Скан": await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь": await help_h(u, c)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Auto-Scan: {'✅ ON' if s['auto_scan'] else '❌ OFF'}", reply_markup=get_main_kb(uid))

# --- SCAN ENGINE (FIXED STOP & LOGIC) ---
async def run_scan(context, uid, s, manual=False, is_auto=False):
    if is_auto:
        last = STATE.sent_signals_cache.get("last_auto_scan_ts")
        if last and (datetime.now() - last).total_seconds() < 1800: return
        STATE.sent_signals_cache["last_auto_scan_ts"] = datetime.now()

    if uid in STATE.abort_scan_users: STATE.abort_scan_users.remove(uid)
    
    ticks = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(ticks)
    filt_txt = "🔥 New" if (s['show_new_only'] or is_auto) else "✅ All"
    tit = f"🔄 Auto" if is_auto else f"🚀 Scan"
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 STOP", callback_data="stop")]])
    status_msg = await context.bot.send_message(chat_id=uid, text=f"{tit}: {filt_txt}\nStarting...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    
    for i in range(total):
        if uid in STATE.abort_scan_users:
            await status_msg.edit_text(f"🛑 Aborted at {i}/{total}.")
            STATE.abort_scan_users.remove(uid)
            return
            
        t = ticks[i]
        res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
        
        if res:
            show = False
            if is_auto:
                if res['Is_New']: show = True
            else:
                if s['show_new_only']: 
                    if res['Is_New']: show = True
                else: show = True
            
            if is_auto and show:
                if res['Ticker'] in STATE.sent_signals_cache["tickers"]: show = False
                else: STATE.sent_signals_cache["tickers"].add(res['Ticker'])

            if show:
                found += 1
                await send_sig(context, uid, res)
        
        if i % 5 == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            filled = int(10 * pct / 100)
            bar = "█" * filled + "░" * (10 - filled)
            try: await status_msg.edit_text(f"{tit}: {filt_txt}\n{pct}% [{bar}] {i+1}/{total}\nFound: {found}", reply_markup=pkb)
            except: pass

    try: 
        await status_msg.edit_text(f"✅ {tit} Complete!\nSignals found: {found}", reply_markup=None)
        if manual: await context.bot.send_message(chat_id=uid, text="Done.", reply_markup=get_main_kb(uid))
    except: pass

async def send_sig(ctx, uid, r):
    ticker = r['Ticker']
    tv_t = ticker.replace('-', '.')
    
    # Prefix fetch
    prefix = ""
    try:
        loop = asyncio.get_running_loop()
        exch = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).fast_info.exchange)
        if exch in ['NMS', 'NGM', 'NCM', 'PNK']: prefix = "NASDAQ:"
        elif exch in ['NYQ', 'NYE']: prefix = "NYSE:"
        elif exch == 'ASE': prefix = "AMEX:"
    except: pass
    
    full_tv = f"{prefix}{tv_t}"
    link = f"https://www.tradingview.com/chart/?symbol={full_tv}"
    ic = "🔥 NEW" if r['Is_New'] else "✅ ACTIVE"
    
    txt = (
        f"{ic} **[{full_tv}]({link})** | **Price**: ${r['Price']:.2f} | **P/E**: {r['PE']}\n"
        f"📊 **ATR**: {r['ATR_Pct']:.2f}% | **SL ATR**: ${r['ATR_SL']:.2f}\n"
        f"🎯 **RR**: {r['RR']:.2f} | 🛑 **SL**: ${r['SL']:.2f}\n"
        f"🏁 **TP**: ${r['TP']:.2f} | 📦 **Size**: {r['Shares']} stocks"
    )
    await ctx.bot.send_message(uid, txt, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

# ==========================================
# 7. AUTO JOB
# ==========================================
async def auto_job(ctx: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    STATE.last_scan = now
    
    today = now.strftime("%Y-%m-%d")
    if STATE.sent_signals_cache["date"] != today:
        STATE.sent_signals_cache.update({"date": today, "tickers": set()})
    
    market_open = (now.weekday() < 5) and (time(9, 30) <= now.time() <= time(16, 0))
    if market_open:
        STATE.add_log(f"🔄 Auto-Scan Started: {now.strftime('%H:%M')}")
        for uid, s in STATE.user_settings.items():
            if s.get('auto_scan', False):
                await run_scan(ctx, uid, s, manual=False, is_auto=True)
    else:
        STATE.add_log(f"💤 Market Hours Check: Closed")

# ==========================================
# 8. STARTUP (SINGLETON)
# ==========================================
@st.cache_resource
def start_bot_singleton():
    if not TG_TOKEN: return None
    
    async def run_bot():
        app = ApplicationBuilder().token(TG_TOKEN).build()
        app.add_handler(CommandHandler('start', start_h))
        app.add_handler(CommandHandler('help', help_h))
        app.add_handler(CallbackQueryHandler(btn_h))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), txt_h))
        
        app.job_queue.run_repeating(auto_job, interval=3600, first=10)
        
        STATE.add_log("🟢 Bot Engine Started")
        await app.run_polling(stop_signals=[], drop_pending_updates=True)

    def loop_in_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_bot())

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop_in_thread, args=(loop,), daemon=True)
    t.start()
    STATE.bot_thread = t
    return t

if __name__ == '__main__':
    if TG_TOKEN:
        start_bot_singleton()
