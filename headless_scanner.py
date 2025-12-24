import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import asyncio
import threading
import pytz
import logging
from datetime import datetime, time, timedelta
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 0. CONFIGURATION & GLOBALS
# ==========================================
TG_TOKEN = "8407386703:AAFtzeEQlc0H2Ev_cccHJGE3mTdxA2c_JkA"
ADMIN_ID = "1335722880"
GITHUB_USERS_URL = "https://raw.githubusercontent.com/vladimirsk1234/vova-screener/refs/heads/main/users.txt"

# --- GLOBAL SHARED STATE (For Dashboard) ---
class BotMonitor:
    status = "OFFLINE"
    approved_users_count = 0
    last_scan_time = "Never"
    next_auto_scan = "Waiting for schedule..."
    total_signals_sent = 0

monitor = BotMonitor()

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

# ==========================================
# 1. EXACT LOGIC & MATH (100% FROM WEB APP)
# ==========================================

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        tickers = [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
        return tickers
    except Exception as e:
        return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# --- INDICATORS ---
def calc_sma(s, l): return s.rolling(l).mean()
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()
def calc_macd(s, f=12, sl=26, sig=9):
    fast = s.ewm(span=f, adjust=False).mean()
    slow = s.ewm(span=sl, adjust=False).mean()
    macd = fast - slow
    return macd - macd.ewm(span=sig, adjust=False).mean()

def calc_adx_pine(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    
    up = h - h.shift(1); down = l.shift(1) - l
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    def rma(s, len): return s.ewm(alpha=1/len, adjust=False).mean()
    
    tr_s = rma(tr, length).replace(0, np.nan)
    p_di = 100 * (rma(pd.Series(p_dm, index=df.index), length) / tr_s)
    m_di = 100 * (rma(pd.Series(m_dm, index=df.index), length) / tr_s)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
    return rma(dx, length), p_di, m_di

def calc_atr(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# --- STRATEGY LOGIC ---
def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    # Indicators
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    # Iterative Structure
    n = len(df)
    c_a, h_a, l_a = df['Close'].values, df['High'].values, df['Low'].values
    
    seq_st = np.zeros(n, dtype=int)
    crit_lvl = np.full(n, np.nan)
    res_peak = np.full(n, np.nan)
    res_struct = np.zeros(n, dtype=bool)
    
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]; s_l = l_a[0]
    
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h
        prev_sl = s_l
        
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1: 
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                pk_hh = is_hh
                last_pk = prev_sh 
                s_state = -1
                s_h = h; s_l = l
                s_crit = h 
            else: 
                is_hl = True if np.isnan(last_tr) else (prev_sl > last_tr)
                tr_hl = is_hl
                last_tr = prev_sl
                s_state = 1
                s_h = h; s_l = l
                s_crit = l 
        else:
            s_state = prev_st
            if s_state == 1: 
                if h >= s_h: s_h = h
                if h >= prev_sh: s_crit = l
                else: s_crit = prev_cr
            elif s_state == -1: 
                if l <= s_l: s_l = l
                if l <= prev_sl: s_crit = h
                else: s_crit = prev_cr
            else: 
                if c > prev_sh: s_state = 1; s_crit = l
                elif c < prev_sl: s_state = -1; s_crit = h
                else: s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        
        seq_st[i] = s_state
        crit_lvl[i] = s_crit
        res_peak[i] = last_pk
        res_struct[i] = (pk_hh and tr_hl)

    # Super Trend
    adx_str = adx >= adx_thr
    bull = (adx_str & (p_di > m_di)) & \
           ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & \
           (efi > 0)
    bear = (adx_str & (m_di > p_di)) & \
           ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & \
           (efi < 0)
           
    t_st = np.zeros(n, dtype=int)
    t_st[bull] = 1
    t_st[bear] = -1
    
    df['Seq'] = seq_st
    df['Crit'] = crit_lvl
    df['Peak'] = res_peak
    df['Struct'] = res_struct
    df['Trend'] = t_st
    df['ATR'] = atr
    
    return df

def analyze_trade(df, idx):
    r = df.iloc[idx]
    errs = []
    
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(r['SMA']) or r['Close'] <= r['SMA']: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(r['Peak']) or np.isnan(r['Crit']): errs.append("NO DATA")
    
    if errs: return False, {}, " ".join(errs)
    
    price = r['Close']
    tp = r['Peak']
    crit = r['Crit']
    atr = r['ATR']
    
    sl_struct = crit
    sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr)
    
    risk = price - final_sl
    reward = tp - price
    
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    rr = reward / risk
    
    return True, {
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": rr, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"

# ==========================================
# 2. BOT LOGIC & STATE
# ==========================================

EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14

# User DB
users_db = {}
scan_history = {} # { 'YYYY-MM-DD': { 'chat_id_TICKER': True } }

DEFAULT_CONFIG = {
    'portfolio': 10000,
    'min_rr': 1.25,
    'risk_pct': 0.2,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True
}

def get_user_state(chat_id):
    if chat_id not in users_db:
        users_db[chat_id] = {
            'config': DEFAULT_CONFIG.copy(),
            'running': False,
            'manual_done': False,
            'auto_active': False,
            'awaiting_input': None,
            'awaiting_tickers': False
        }
    return users_db[chat_id]

async def check_auth(user_id, username):
    if str(user_id) == ADMIN_ID: return True
    try:
        resp = requests.get(GITHUB_USERS_URL)
        if resp.status_code == 200:
            allowed = [u.strip() for u in resp.text.splitlines() if u.strip()]
            monitor.approved_users_count = len(allowed)
            return username in allowed
    except:
        return False
    return False

def is_market_open():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    # Market Open calculation for Dashboard
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now > market_open: market_open += timedelta(days=1)
    
    diff = market_open - now
    monitor.next_auto_scan = f"In {int(diff.total_seconds()//60)} mins (if active)"
    
    if now.weekday() > 4: 
        monitor.next_auto_scan = "Market Closed (Weekend)"
        return False 
    
    open_t = time(9, 30)
    close_t = time(16, 0)
    is_open = open_t <= now.time() <= close_t
    
    if not is_open:
        monitor.next_auto_scan = "Market Closed"
    
    return is_open

def format_card(t, d, shares, is_new, pe_val):
    tv_link = f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}"
    
    new_icon = "🆕" if is_new else ""
    header = f"💎 <a href='{tv_link}'><b>{t}</b></a> {new_icon}"
    
    pe_str = f"P/E {pe_val:.0f}" if pe_val else ""
    price_line = f"💵 <b>${d['P']:.2f}</b> {pe_str}"
    
    val_pos = shares * d['P']
    profit = (d['TP'] - d['P']) * shares
    loss = (d['P'] - d['SL']) * shares
    
    # Compact Luxury Layout
    card = (
        f"{header}\n"
        f"{price_line}\n"
        f"<code>──────────────</code>\n"
        f"⚖️ RR: <b>{d['RR']:.2f}</b> | 💼 Pos: <b>{shares}</b> (${val_pos:.0f})\n"
        f"🎯 TP: <b>{d['TP']:.2f}</b> (<i style='color:green'>+${profit:.0f}</i>)\n"
        f"🛑 SL: <b>{d['SL']:.2f}</b> (<i style='color:red'>-${loss:.0f}</i>)\n"
        f"📊 ATR: {d['ATR']:.2f} | 🧱 Crit: {d['Crit']:.2f}"
    )
    return card

# ==========================================
# 3. MENUS (PHYSICAL BUTTONS)
# ==========================================

def get_main_menu(cfg, auto_active):
    auto_icon = "🟢" if auto_active else "🔴"
    
    # Main Dashboard Style Buttons
    row1 = [KeyboardButton("▶ START SCAN"), KeyboardButton("⏹ STOP SCAN")]
    row2 = [KeyboardButton(f"💰 Port: ${cfg['portfolio']}"), KeyboardButton(f"⚖️ RR: {cfg['min_rr']}")]
    row3 = [KeyboardButton(f"⚠️ Risk: {cfg['risk_pct']}%"), KeyboardButton(f"📊 ATR: {cfg['max_atr']}%")]
    row4 = [KeyboardButton(f"📈 SMA: {cfg['sma']}"), KeyboardButton(f"📅 TF: {cfg['tf']}")]
    row5 = [KeyboardButton(f"🆕 New Only: {cfg['new_only']}"), KeyboardButton(f"🔄 AUTO: {auto_icon}")]
    row6 = [KeyboardButton("🔎 CHECK TICKERS")]
    
    return ReplyKeyboardMarkup([row1, row2, row3, row4, row5, row6], resize_keyboard=True)

# ==========================================
# 4. SCANNER CORE
# ==========================================
async def scan_task(context, chat_id, tickers, mode="Manual"):
    state = get_user_state(chat_id)
    
    # 1. FREEZE PARAMS (Safety feature: changing params during scan won't affect running scan)
    p = state['config'].copy()
    
    prog_msg = await context.bot.send_message(chat_id, f"🚀 <b>Starting {mode} Scan...</b>\nTarget: {len(tickers)} tickers", parse_mode=ParseMode.HTML)
    
    found = 0
    today = datetime.now().strftime('%Y-%m-%d')
    if today not in scan_history: scan_history[today] = {}
    
    monitor.last_scan_time = datetime.now().strftime("%H:%M:%S UTC")

    for i, t in enumerate(tickers):
        # Stop Check (Only for manual loops)
        if mode == "Manual" and not state['running']:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=prog_msg.message_id, text="⏹ <b>Scan Stopped by User.</b>", parse_mode=ParseMode.HTML)
            return

        # Progress Bar Update (Every 5% or 10 tickers)
        if mode == "Manual" and (i % 10 == 0 or i == len(tickers)-1):
            pct = int((i + 1) / len(tickers) * 100)
            bar = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
            try:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=prog_msg.message_id, text=f"🔍 <b>Scanning...</b> {pct}%\n{bar}\nChecking: {t}", parse_mode=ParseMode.HTML)
            except: pass

        # Auto Duplicate Check
        if mode == "Auto":
            if f"{chat_id}_{t}" in scan_history[today]: continue

        try:
            # Data Fetch
            inter = "1d" if p['tf'] == "Daily" else "1wk"
            per = "2y" if p['tf'] == "Daily" else "5y"
            
            # Threaded fetch to not block bot loop
            df = await asyncio.to_thread(yf.download, t, period=per, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            valid = True
            reason = ""
            
            if len(df) < p['sma'] + 5:
                valid = False; reason = "NO DATA"
            else:
                df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
                valid, d, reason = analyze_trade(df, -1)
            
            # Report rejection only if specific ticker scan
            if mode == "Single" and not valid:
                await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: {reason}", parse_mode=ParseMode.HTML)
                continue

            if not valid: continue

            # Filters
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if p['new_only'] and not is_new: continue

            if d['RR'] < p['min_rr']:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ Low RR: {d['RR']:.2f}", parse_mode=ParseMode.HTML)
                continue
                
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > p['max_atr']:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ High ATR: {atr_pct:.1f}%", parse_mode=ParseMode.HTML)
                continue

            # Position
            risk_amt = p['portfolio'] * (p['risk_pct'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            
            shares = int(risk_amt / risk_share)
            max_shares = int(p['portfolio'] / d['P'])
            shares = min(shares, max_shares)
            
            if shares < 1:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ Low Funds", parse_mode=ParseMode.HTML)
                continue

            # Success!
            pe = await asyncio.to_thread(get_financial_info, t)
            card = format_card(t, d, shares, is_new, pe)
            
            await context.bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            found += 1
            monitor.total_signals_sent += 1
            
            if mode == "Auto":
                scan_history[today][f"{chat_id}_{t}"] = True

        except Exception as e:
            continue

    state['running'] = False
    if mode == "Manual":
        state['manual_done'] = True
        await context.bot.edit_message_text(chat_id=chat_id, message_id=prog_msg.message_id, text=f"🏁 <b>Scan Complete.</b> Found: {found}", parse_mode=ParseMode.HTML)

# ==========================================
# 5. HANDLERS
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not await check_auth(user.id, user.username):
        await update.message.reply_text("⛔ Access Denied.")
        return
    
    st = get_user_state(update.effective_chat.id)
    await update.message.reply_text(f"💎 <b>VOVA SCREENER READY</b>\nPhysical buttons initialized.", 
                                    parse_mode=ParseMode.HTML,
                                    reply_markup=get_main_menu(st['config'], st['auto_active']))

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not await check_auth(user.id, user.username): return
    
    text = update.message.text
    chat_id = update.effective_chat.id
    st = get_user_state(chat_id)
    cfg = st['config']

    # --- INPUT CAPTURE ---
    if st['awaiting_input']:
        key = st['awaiting_input']
        try:
            if key == 'portfolio': val = int(text)
            elif key in ['min_rr', 'risk_pct', 'max_atr']: val = float(text)
            else: val = text
            
            cfg[key] = val
            st['awaiting_input'] = None
            await update.message.reply_text(f"✅ Set {key} to {val}", reply_markup=get_main_menu(cfg, st['auto_active']))
            return
        except:
            await update.message.reply_text("❌ Invalid format. Try again.", reply_markup=get_main_menu(cfg, st['auto_active']))
            return

    # --- INPUT TICKERS ---
    if st['awaiting_tickers']:
        tickers = [x.strip().upper() for x in text.split(',') if x.strip()]
        st['awaiting_tickers'] = False
        asyncio.create_task(scan_task(context, chat_id, tickers, "Single"))
        await update.message.reply_text("🔎 Analyzing...", reply_markup=get_main_menu(cfg, st['auto_active']))
        return

    # --- MENU ACTIONS ---
    if text == "▶ START SCAN":
        st['manual_done'] = True # Trigger requirement for auto
        st['running'] = True
        tickers = get_sp500_tickers()
        if tickers:
            asyncio.create_task(scan_task(context, chat_id, tickers, "Manual"))
        else:
            st['running'] = False
            await update.message.reply_text("❌ S&P Data Error")
    
    elif text == "⏹ STOP SCAN":
        st['running'] = False
        await update.message.reply_text("🛑 Stopping...", reply_markup=get_main_menu(cfg, st['auto_active']))

    elif text == "🔎 CHECK TICKERS":
        st['awaiting_tickers'] = True
        await update.message.reply_text("⌨️ Type tickers separated by comma (e.g., AAPL, TSLA):")

    elif "🔄 AUTO" in text:
        if not st['manual_done']:
            await update.message.reply_text("⚠️ You must perform at least ONE manual scan (press Start Scan) before enabling Auto.")
        else:
            st['auto_active'] = not st['auto_active']
            await update.message.reply_text(f"Auto Scan: {st['auto_active']}", reply_markup=get_main_menu(cfg, st['auto_active']))

    # --- SETTINGS TOGGLES & INPUT TRIGGERS ---
    elif "SMA:" in text:
        curr = cfg['sma']
        nxt = 150 if curr == 100 else (200 if curr == 150 else 100)
        cfg['sma'] = nxt
        await update.message.reply_text(f"SMA set to {nxt}", reply_markup=get_main_menu(cfg, st['auto_active']))
    
    elif "TF:" in text:
        curr = cfg['tf']
        nxt = "Weekly" if curr == "Daily" else "Daily"
        cfg['tf'] = nxt
        await update.message.reply_text(f"Timeframe set to {nxt}", reply_markup=get_main_menu(cfg, st['auto_active']))
        
    elif "New Only:" in text:
        cfg['new_only'] = not cfg['new_only']
        await update.message.reply_text(f"New Only set to {cfg['new_only']}", reply_markup=get_main_menu(cfg, st['auto_active']))

    elif "Port:" in text:
        st['awaiting_input'] = 'portfolio'
        await update.message.reply_text("🔢 Type Portfolio Size ($):")
        
    elif "RR:" in text:
        st['awaiting_input'] = 'min_rr'
        await update.message.reply_text("🔢 Type Min RR (e.g. 1.5):")
        
    elif "Risk:" in text:
        st['awaiting_input'] = 'risk_pct'
        await update.message.reply_text("🔢 Type Risk % (e.g. 0.5):")
        
    elif "ATR:" in text:
        st['awaiting_input'] = 'max_atr'
        await update.message.reply_text("🔢 Type Max ATR % (e.g. 4.0):")

# ==========================================
# 6. AUTO JOB
# ==========================================
async def hourly_job(context: ContextTypes.DEFAULT_TYPE):
    # Runs every hour
    if not is_market_open(): return
    
    tickers = get_sp500_tickers()
    if not tickers: return
    
    for cid, data in users_db.items():
        if data['auto_active'] and data['manual_done']:
            asyncio.create_task(scan_task(context, cid, tickers, "Auto"))

# ==========================================
# 7. MAIN THREAD RUNNER
# ==========================================
def run_bot():
    import nest_asyncio
    nest_asyncio.apply()
    
    monitor.status = "ONLINE"
    
    app = Application.builder().token(TG_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, message_handler))
    
    # Auto scan job
    app.job_queue.run_repeating(hourly_job, interval=3600, first=10)
    
    # FIX FOR STREAMLIT CLOUD RUNTIME ERROR
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None, close_loop=False)

# ==========================================
# 8. STREAMLIT DASHBOARD (SERVER SIDE)
# ==========================================

st.set_page_config(page_title="Vova Bot Server", layout="centered", page_icon="🤖")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41444C;
        text-align: center;
    }
    .big-stat { font-size: 24px; font-weight: bold; color: #00E676; }
    .label { font-size: 14px; color: #A0A0A0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Vova Screener Bot Server")

# Auto refresh dashboard every 10 seconds to show updates
st_autorefresh(interval=10000, key="datarefresh")

# Background Thread Start
if 'bot_thread' not in st.session_state:
    st.session_state.bot_thread = threading.Thread(target=run_bot, daemon=True)
    st.session_state.bot_thread.start()

# Dashboard Grid
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">STATUS</div>
        <div class="big-stat" style="color: {'#00E676' if monitor.status=='ONLINE' else 'red'}">{monitor.status}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">APPROVED USERS</div>
        <div class="big-stat">{monitor.approved_users_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

c3, c4 = st.columns(2)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">LAST SCAN</div>
        <div class="big-stat" style="color: #448AFF">{monitor.last_scan_time}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    # Update market status logic calculation
    is_market_open() 
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">NEXT AUTO SCAN</div>
        <div class="big-stat" style="color: #FFAB00">{monitor.next_auto_scan}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption(f"Total Signals Processed Since Start: {monitor.total_signals_sent}")
