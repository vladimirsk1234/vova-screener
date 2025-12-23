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
# 3. INTERFACE
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
                    c1.metric("✅ Users", f"{len(ul)}")
                else: c1.error(f"GH Error: {r.status_code}")
            except: c1.error("Net Error")
        else: c1.warning("No GH URL")
            
        bs = "🟢 Active" if (STATE.bot_thread and STATE.bot_thread.is_alive()) else "🔴 Stopped"
        c2.metric("Bot Status", bs)

        st.subheader("🕒 Auto-Scan Status")
        ct1, ct2 = st.columns(2)
        if STATE.last_scan:
            ny = pytz.timezone('US/Eastern')
            ls = STATE.last_scan if STATE.last_scan.tzinfo else ny.localize(STATE.last_scan)
            ct1.metric("Last Run (NY)", ls.astimezone(ny).strftime("%H:%M:%S"))
            
            nxt = ls + timedelta(hours=1)
            rem = nxt - datetime.now(ls.tzinfo)
            if rem.total_seconds() > 0:
                m, s = divmod(int(rem.total_seconds()), 60)
                ct2.metric("Next Run In", f"{m}m {s}s")
            else: ct2.metric("Next Run", "Running...")
        else:
            ct1.metric("Last Run", "Waiting...")
            ct2.metric("Next Run", "Soon...")

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
# 5. STRATEGY
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

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
            "Is_New": (v_tod and not v_yest)
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
        await u.message.reply_text(f"⛔ Access Denied. ID: `{uid}`")
        return
    await u.message.reply_text("👋 **Vova Screener Bot**", reply_markup=get_main_kb(uid), parse_mode=ParseMode.MARKDOWN)

async def help_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = (
        "ℹ️ **Info**\n"
        "Strat: BoS + SuperTrend (S&P 500).\n\n"
        "🛠 **Settings:**\n"
        "• Port: Deposit\n"
        "• Risk %: Risk/Trade\n"
        "• RR: Reward/Risk\n"
        "🔄 **Auto:** Hourly (9:30-16:00 ET). New only.\n"
        "🚀 **Manual:** Check now."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Settings:**\n"
        f"💰 ${s['portfolio_size']:,} | ⚠️ {s['risk_per_trade_pct']}%\n"
        f"📊 RR: {s['min_rr']} | 🔍 {s['scan_mode']}\n"
        f"👀 {'🔥 New' if s['show_new_only'] else '✅ All'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Port: ${s['portfolio_size']} ✏️", callback_data="ask_port")],
        [InlineKeyboardButton(f"Mode: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥' if s['show_new_only'] else '✅'} 🔄", callback_data="ch_filt")]
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
        await q.message.reply_text("🛑 Stopping...")
        return

    if d.startswith("ask_"):
        STATE.user_states[uid] = d.split("_")[1].upper()
        await q.message.reply_text(f"Enter Value:", parse_mode=ParseMode.MARKDOWN)
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
            await u.message.reply_text("Cancel.", reply_markup=get_main_kb(uid))
        else:
            try:
                val = float(txt.replace(',', '.').replace('%', '').replace('$', ''))
                s = get_settings(uid)
                if st_code == "RISK": s['risk_per_trade_pct'] = val
                elif st_code == "RR": s['min_rr'] = val
                elif st_code == "PORT": s['portfolio_size'] = int(val)
                del STATE.user_states[uid]
                await u.message.reply_text(f"✅ Saved: {val}")
                await settings_menu(u, c)
                return
            except:
                await u.message.reply_text("❌ Number Error.")
                return

    if txt == "🚀 Запустить Скан": await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь": await help_h(u, c)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Auto: {'✅ ON' if s['auto_scan'] else '❌ OFF'}", reply_markup=get_main_kb(uid))

# --- SCAN ENGINE ---
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
    status = await context.bot.send_message(chat_id=uid, text=f"{tit}: {filt_txt}\nStarting...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    batch_sz = 5
    
    for i in range(0, total, batch_sz):
        if uid in STATE.abort_scan_users:
            await status.edit_text(f"🛑 Stopped at {i}/{total}.")
            STATE.abort_scan_users.remove(uid)
            return
            
        batch = ticks[i:i+batch_sz]
        for t in batch:
            if uid in STATE.abort_scan_users: break
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
        
        pct = int((i+len(batch))/total*100)
        filled = int(10 * pct / 100)
        bar = "█"*filled + "░"*(10-filled)
        try: await status.edit_text(f"{tit}: {filt_txt}\n{pct}% [{bar}] {i+len(batch)}/{total}\nFound: {found}", reply_markup=pkb)
        except: pass

    try: 
        await status.edit_text(f"✅ {tit} Done!\nFound: {found}", reply_markup=None)
        if manual: await context.bot.send_message(chat_id=uid, text="Menu:", reply_markup=get_main_kb(uid))
    except: pass

async def send_sig(ctx, uid, r):
    tv = r['Ticker'].replace('-', '.')
    link = f"https://www.tradingview.com/chart/?symbol={tv}"
    ic = "🔥 NEW" if r['Is_New'] else "✅ ACTIVE"
    txt = (
        f"{ic} **[{tv}]({link})** | ${r['Price']:.2f}\n"
        f"📊 ATR: {r['ATR_Pct']:.2f}% | SL: ${r['ATR_SL']:.2f}\n"
        f"🎯 RR: {r['RR']:.2f} | 🛑 SL: ${r['SL']:.2f}\n"
        f"🏁 TP: ${r['TP']:.2f} | 📦 Size: {r['Shares']}"
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
        STATE.add_log(f"🔄 Auto-Scan: {now.strftime('%H:%M')}")
        for uid, s in STATE.user_settings.items():
            if s.get('auto_scan', False):
                await run_scan(ctx, uid, s, manual=False, is_auto=True)
    else:
        STATE.add_log(f"💤 Market Closed")

# ==========================================
# 8. STARTUP (FIXED SINGLETON)
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
        
        job_q = app.job_queue
        job_q.run_repeating(auto_job, interval=3600, first=10)
        
        STATE.add_log("🟢 Polling Started")
        # RUN POLLING INSTEAD OF MANUAL UPDATER START
        # This handles initialization and start automatically
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
