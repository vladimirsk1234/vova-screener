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
# 2. НАСТРОЙКИ
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
# 3. INTERFACE (STREAMLIT)
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
# 4. HELPERS
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

def calc_sma(s, l): return s.rolling(window=l).mean()
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()
def calc_atr(df, l):
    h, lo, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/l, adjust=False).mean()
def calc_macd(s, f=12, sl=26, sig=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es
    si = m.ewm(span=sig, adjust=False).mean()
    return m, si, m - si
def calc_adx(df, l):
    h, lo, c = df['High'], df['Low'], df['Close']
    u, d = h - h.shift(1), lo.shift(1) - lo
    p_dm = np.where((u > d) & (u > 0), u, 0.0)
    m_dm = np.where((d > u) & (d > 0), d, 0.0)
    a = 1.0/l
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_s = tr.ewm(alpha=a, adjust=False).mean().replace(0, np.nan)
    p_s = pd.Series(p_dm, index=df.index).ewm(alpha=a, adjust=False).mean()
    m_s = pd.Series(m_dm, index=df.index).ewm(alpha=a, adjust=False).mean()
    p_di = 100 * (p_s / tr_s)
    m_di = 100 * (m_s / tr_s)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di)
    return dx.ewm(alpha=a, adjust=False).mean(), p_di, m_di

# ==========================================
# 5. STRATEGY (100% LOGIC)
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        # Fetch P/E
        pe = t_obj.info.get('trailingPE', 'N/A')
        if pe != 'N/A': pe = f"{pe:.2f}"

        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_s, pdi, mdi = calc_adx(df, settings['adx_len'])
        atr_s = calc_atr(df, settings['atr_len'])
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        df['EFI'] = calc_ema(df['Close'].diff() * df['Volume'], settings['len_fast'])

        c_arr, h_arr, l_arr = df['Close'].values, df['High'].values, df['Low'].values
        ema_f, ema_s = df['EMA_Fast'].values, df['EMA_Slow'].values
        h_vals, efi_vals = macd_hist.values, df['EFI'].values
        adx_v, pdi_v, mdi_v = adx_s.values, pdi.values, mdi.values

        n = len(df)
        t_lst, s_lst, crit_lst, peak_lst, struct_lst = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n
        seq_st, crit, s_h, s_l = 0, np.nan, h_arr[0], l_arr[0]
        l_peak, l_trough, l_hh, l_hl = np.nan, np.nan, False, False

        for i in range(1, n):
            c, h, l = c_arr[i], h_arr[i], l_arr[i]
            prev_st, is_brk = seq_st, False
            if prev_st == 1 and not np.isnan(crit): is_brk = c < crit
            elif prev_st == -1 and not np.isnan(crit): is_brk = c > crit
            
            if is_brk:
                if prev_st == 1:
                    is_hh = True if np.isnan(l_peak) else (s_h > l_peak)
                    l_hh, l_peak = is_hh, s_h
                    seq_st, s_h, s_l, crit = -1, h, l, h
                else:
                    is_hl = True if np.isnan(l_trough) else (s_l > l_trough)
                    l_hl, l_trough = is_hl, s_l
                    seq_st, s_h, s_l, crit = 1, h, l, l
            else:
                if seq_st == 1:
                    if h >= s_h: s_h = h
                    if h >= s_h: crit = l
                elif seq_st == -1:
                    if l <= s_l: s_l = l
                    if l <= s_l: crit = h
                else:
                    if c > s_h: seq_st, crit = 1, l
                    elif c < s_l: seq_st, crit = -1, h
                    else: s_h, s_l = max(s_h, h), min(s_l, l)

            strong = (adx_v[i] > settings['adx_thresh'])
            rising = (ema_f[i] > ema_f[i-1]) and (ema_s[i] > ema_s[i-1])
            falling = (ema_f[i] < ema_f[i-1]) and (ema_s[i] < ema_s[i-1])
            bull = strong and (pdi_v[i] > mdi_v[i])
            bear = strong and (mdi_v[i] > pdi_v[i])
            
            curr_t = 0
            if bull and rising and (h_vals[i] > h_vals[i-1]) and (efi_vals[i] > 0): curr_t = 1
            elif bear and falling and (h_vals[i] < h_vals[i-1]) and (efi_vals[i] < 0): curr_t = -1
            
            t_lst[i], s_lst[i], crit_lst[i], peak_lst[i], struct_lst[i] = curr_t, seq_st, crit, l_peak, (l_hh and l_hl)

        def check(idx):
            if idx >= n or idx < 0: return False, 0.0, np.nan, np.nan
            p, sma = c_arr[idx], df['SMA_Major'].iloc[idx]
            valid = (s_lst[idx] == 1) and ((p > sma) if not np.isnan(sma) else False) and (t_lst[idx] != -1) and struct_lst[idx]
            rr, cr, pk = 0.0, crit_lst[idx], peak_lst[idx]
            if valid and not np.isnan(pk) and not np.isnan(cr):
                rsk, rwd = p - cr, pk - p
                if rsk > 0 and rwd > 0: rr = rwd / rsk
                else: valid = False
            return valid, rr, cr, pk

        v_tod, rr_t, sl_t, tp_t = check(n-1)
        v_yest, _, _, _ = check(n-2)
        
        if not v_tod: return None
        if rr_t < settings['min_rr']: return None
        
        cur_c = c_arr[-1]
        cur_atr = atr_s.iloc[-1]
        atr_pct = (cur_atr / cur_c) * 100
        if atr_pct > settings['max_atr_pct']: return None
        
        rsk_sh = cur_c - sl_t
        shares = 0
        if rsk_sh > 0:
            rsk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
            shares = int(rsk_amt / rsk_sh)
            shares = min(shares, int(settings['portfolio_size'] / cur_c))
            if shares < 1: shares = 1

        return {
            "Ticker": ticker, "Price": cur_c, "RR": rr_t, "SL": sl_t, "TP": tp_t,
            "ATR_SL": cur_c - cur_atr, "Shares": shares, "ATR_Pct": atr_pct, 
            "Is_New": (v_tod and not v_yest), "PE": pe
        }
    except: return None

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
        await u.message.reply_text(f"⛔ Access Denied.\n\nYour Telegram ID: `{uid}`\n\nSend this ID to @Vova_Skl to get access.", parse_mode=ParseMode.MARKDOWN)
        return
    
    welcome = (
        "👋 **Welcome to Vova Screener Bot!**\n\n"
        "I use **Vova Strategy** (Structure Break + Trends) to find high-probability US stock setups.\n\n"
        "🛠 **How to use:**\n"
        "1. Click **Scan** to check market manually.\n"
        "2. Use **Settings** to adjust risk and filters.\n"
        "3. Turn on **Auto-Scan** for hourly notifications.\n\n"
        "Use ℹ️ **Help** for detailed explanation."
    )
    await u.message.reply_text(welcome, reply_markup=get_main_kb(uid), parse_mode=ParseMode.MARKDOWN)

async def help_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = (
        "ℹ️ **Vova Screener Bot Manual**\n\n"
        "**Strategy Explanation:**\n"
        "The bot looks for **Bullish Structure Breaks** where a stock has established higher highs/lows and stays above SMA 200. It also checks for momentum using SuperTrend filters (EMA, MACD Histogram, ADX Strength).\n\n"
        "**Parameter Descriptions:**\n"
        "💰 **Portfolio Size**: Your total trading capital. Used to calculate how many stocks to buy.\n"
        "⚠️ **Risk %**: How much of your portfolio you're willing to lose on **one** trade if Stop Loss is hit.\n"
        "📊 **Min RR**: Minimum Risk/Reward ratio. If target is too close compared to risk, signal is ignored.\n"
        "📈 **Max ATR %**: Filters out stocks that are too volatile (moving too many % per day).\n\n"
        "**Automatic Mode:**\n"
        "When 🔄 **Auto** is ✅, the bot scans **S&P 500** every 1 hour while the US market is open (9:30 AM - 4:00 PM ET). It only alerts you about **New** signals."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Bot Configuration:**\n\n"
        f"💰 **Portfolio**: ${s['portfolio_size']:,}\n"
        f"⚠️ **Risk Per Trade**: {s['risk_per_trade_pct']}%\n"
        f"📊 **Minimum RR**: {s['min_rr']}\n"
        f"📈 **Volatility Filter (Max ATR)**: {s['max_atr_pct']}%\n"
        f"🔍 **Market Mode**: {s['scan_mode']}\n"
        f"👀 **Manual Filter**: {'🔥 Only New' if s['show_new_only'] else '✅ Show All Active'}\n\n"
        f"Select a parameter to edit:"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port"),
         InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Market: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥 New' if s['show_new_only'] else '✅ All'} 🔄", callback_data="ch_filt")],
        [InlineKeyboardButton("ℹ️ HELP / STRATEGY INFO", callback_data="show_help")]
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
        await q.message.reply_text("🛑 User request: Aborting scan...")
        return

    if d == "show_help":
        await help_h(u, c)
        return

    if d.startswith("ask_"):
        STATE.user_states[uid] = d.split("_")[1].upper()
        await q.message.reply_text(f"Please type the new numeric value for **{STATE.user_states[uid]}**:", parse_mode=ParseMode.MARKDOWN)
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
            await u.message.reply_text("Input cancelled.", reply_markup=get_main_kb(uid))
        else:
            try:
                val = float(txt.replace(',', '.').replace('%', '').replace('$', '').strip())
                s = get_settings(uid)
                if st_code == "RISK": s['risk_per_trade_pct'] = val
                elif st_code == "RR": s['min_rr'] = val
                elif st_code == "PORT": s['portfolio_size'] = int(val)
                elif st_code == "ATR": s['max_atr_pct'] = val
                del STATE.user_states[uid]
                await u.message.reply_text(f"✅ Parameter updated to: {val}")
                await settings_menu(u, c)
                return
            except:
                await u.message.reply_text("❌ Error: Please enter a valid number.")
                return

    if txt == "🚀 Запустить Скан": await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь": await help_h(u, c)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Automatic Scanner: {'✅ ON' if s['auto_scan'] else '❌ OFF'}", reply_markup=get_main_kb(uid))

# --- SCAN ENGINE (FIXED STOP LOGIC) ---
async def run_scan(context, uid, s, manual=False, is_auto=False):
    # Auto scan frequency check
    if is_auto:
        last = STATE.sent_signals_cache.get("last_auto_scan_ts")
        if last and (datetime.now() - last).total_seconds() < 1800: return
        STATE.sent_signals_cache["last_auto_scan_ts"] = datetime.now()

    # Clear previous stop requests
    if uid in STATE.abort_scan_users: STATE.abort_scan_users.remove(uid)
    
    ticks = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(ticks)
    filt_txt = "🔥 New Signals" if (s['show_new_only'] or is_auto) else "✅ All Valid"
    tit = f"🔄 Auto-Scan" if is_auto else f"🚀 Manual Scan"
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 STOP SCAN", callback_data="stop")]])
    status_msg = await context.bot.send_message(chat_id=uid, text=f"{tit}\nMode: {filt_txt}\nProcessing...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    
    for i in range(total):
        # CRITICAL STOP CHECK (Checks EVERY ticker)
        if uid in STATE.abort_scan_users:
            await status_msg.edit_text(f"🛑 Scanning stopped by user at {i}/{total}.")
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
        
        # UI Update every 5 tickers
        if i % 5 == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            filled = int(10 * pct / 100)
            bar = "█" * filled + "░" * (10 - filled)
            try: 
                await status_msg.edit_text(
                    f"{tit}\n{pct}% [{bar}] {i + 1}/{total}\nFound signals: {found}", 
                    reply_markup=pkb
                )
            except: pass

    fin_txt = f"✅ {tit} Complete!\nSignals found: {found}"
    try: 
        await status_msg.edit_text(fin_txt, reply_markup=None)
        # Always return main menu after manual scan
        if manual: 
            await context.bot.send_message(chat_id=uid, text="Scan finished. Ready for next task.", reply_markup=get_main_kb(uid))
    except: pass

async def send_sig(ctx, uid, r):
    ticker = r['Ticker']
    tv_ticker = ticker.replace('-', '.')
    
    prefix = ""
    try:
        loop = asyncio.get_running_loop()
        exch = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).fast_info.exchange)
        if exch in ['NMS', 'NGM', 'NCM', 'PNK']: prefix = "NASDAQ:"
        elif exch in ['NYQ', 'NYE']: prefix = "NYSE:"
        elif exch == 'ASE': prefix = "AMEX:"
    except: pass
    
    full_tv = f"{prefix}{tv_ticker}"
    link = f"https://www.tradingview.com/chart/?symbol={full_tv}"
    ic = "🔥 NEW" if r['Is_New'] else "✅ ACTIVE"
    
    txt = (
        f"{ic} **[{full_tv}]({link})** | **Price**: ${r['Price']:.2f} | **P/E**: {r['PE']}\n"
        f"📊 **ATR**: {r['ATR_Pct']:.2f}% | **SL ATR**: ${r['ATR_SL']:.2f}\n"
        f"🎯 **RR**: {r['RR']:.2f} | 🛑 **SL**: ${r['SL']:.2f}\n"
        f"🏁 **TP**: ${r['TP']:.2f} | 📦 **Trade Size**: {r['Shares']} shares"
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
        STATE.add_log(f"🔄 Auto-Scan Triggered: {now.strftime('%H:%M')}")
        # Run for all users who have auto scan ON
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
        
        # Schedule hourly auto-scans
        app.job_queue.run_repeating(auto_job, interval=3600, first=10)
        
        STATE.add_log("🟢 Polling Service Started")
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
