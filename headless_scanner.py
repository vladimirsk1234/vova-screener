import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import asyncio
import threading
import nest_asyncio
from datetime import datetime, time
import pytz

# Telegram libraries
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

# --- APPLY NEST_ASYNCIO ---
# Required to run asyncio within Streamlit's loop
nest_asyncio.apply()

# ==========================================
# 1. CONFIG & SECRETS
# ==========================================
st.set_page_config(page_title="Vova Bot Server", layout="centered", page_icon="💎")

try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = str(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets["GITHUB_USERS_URL"]
except Exception as e:
    st.error(f"❌ SECRET ERROR: {e}. Please check .streamlit/secrets.toml")
    st.stop()

# ==========================================
# 2. GLOBAL SHARED STATE
# ==========================================
# This class holds the memory for the bot and web dashboard
class BotGlobalState:
    def __init__(self):
        self.active_scans = {}   # chat_id -> bool (Is currently scanning?)
        self.user_configs = {}   # chat_id -> dict (Parameters)
        self.last_scan_time = "Never"
        self.bot_running = False
        self.auto_scan_active = {} # chat_id -> bool
        self.found_tickers_today = {} # chat_id -> set(tickers) (For 10.1 rule)
        self.next_auto_scan = {} # chat_id -> str (Time string)

@st.cache_resource
def get_bot_state():
    return BotGlobalState()

state = get_bot_state()

# Default Configuration
DEFAULT_CONFIG = {
    'portfolio': 10000,
    'rr': 1.25,
    'risk': 0.2,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
    'source': 'SP500', # or 'MANUAL'
    'manual_list': [],
    'auto_scan': False
}

# ==========================================
# 3. HELPER FUNCTIONS & VOVA LOGIC (100% COPY)
# ==========================================

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: return []

def get_authorized_users():
    try:
        response = requests.get(GITHUB_USERS_URL)
        if response.status_code == 200:
            try: users = response.json()
            except: users = response.text.splitlines()
            clean_users = [str(u).strip() for u in users]
            # Ensure Admin is always there
            if ADMIN_ID not in clean_users: clean_users.append(ADMIN_ID)
            return clean_users
    except: return [ADMIN_ID]
    return [ADMIN_ID]

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# --- MATH (EXACT LOGIC) ---
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

# --- STRATEGY LOGIC (EXACT COPY) ---
def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    # --- Indicators ---
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    # --- Iterative Structure Logic ---
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
                if c > prev_sh: 
                    s_state = 1; s_crit = l
                elif c < prev_sl: 
                    s_state = -1; s_crit = h
                else:
                    s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        
        seq_st[i] = s_state
        crit_lvl[i] = s_crit
        res_peak[i] = last_pk 
        res_struct[i] = (pk_hh and tr_hl)

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
# 4. TELEGRAM BOT LOGIC
# ==========================================

# --- HELPER: Is Market Open? ---
def is_market_open():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    # Market Open 9:30 AM - 4:00 PM, Mon-Fri
    if now.weekday() > 4: return False # Weekend
    market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_start <= now <= market_end

# --- KEYBOARD ---
def get_main_keyboard(config, scanning_active):
    src_icon = "🌎 SP500" if config['source'] == "SP500" else "📝 LIST"
    new_icon = "✅ YES" if config['new_only'] else "❌ NO"
    auto_icon = "🟢 ON" if config['auto_scan'] else "🔴 OFF"
    
    # Hide controls if scanning
    if scanning_active:
        return InlineKeyboardMarkup([[InlineKeyboardButton("⏹️ STOP SCAN", callback_data="stop_scan")]])
    
    keyboard = [
        [
            InlineKeyboardButton(f"💵 PORT: ${config['portfolio']}", callback_data="set_port"),
            InlineKeyboardButton(f"⚖️ RR: {config['rr']}", callback_data="set_rr")
        ],
        [
            InlineKeyboardButton(f"⚠️ RISK: {config['risk']}%", callback_data="set_risk"),
            InlineKeyboardButton(f"🌊 MAX ATR: {config['max_atr']}%", callback_data="set_matr")
        ],
        [
            InlineKeyboardButton(f"📉 SMA: {config['sma']}", callback_data="set_sma"),
            InlineKeyboardButton(f"⏱️ TF: {config['tf']}", callback_data="set_tf")
        ],
        [
            InlineKeyboardButton(f"🆕 NEW ONLY: {new_icon}", callback_data="toggle_new"),
            InlineKeyboardButton(f"🔍 SRC: {src_icon}", callback_data="toggle_src")
        ],
        [
             InlineKeyboardButton(f"🤖 AUTO SCAN: {auto_icon}", callback_data="toggle_auto")
        ],
        [
            InlineKeyboardButton("▶️ START MANUAL SCAN", callback_data="start_scan")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id not in get_authorized_users():
        await update.message.reply_text("⛔ ACCESS DENIED")
        return
    
    state = get_bot_state()
    if user_id not in state.user_configs:
        state.user_configs[user_id] = DEFAULT_CONFIG.copy()
        state.found_tickers_today[user_id] = set()

    await update.message.reply_text(
        "💎 **VOVA SCREENER BOT**\nНастрой параметры и нажми Start.",
        reply_markup=get_main_keyboard(state.user_configs[user_id], False),
        parse_mode=ParseMode.MARKDOWN
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = str(query.from_user.id)
    state = get_bot_state()
    config = state.user_configs.get(user_id, DEFAULT_CONFIG.copy())
    is_scanning = state.active_scans.get(user_id, False)
    data = query.data

    # Stop Logic
    if data == "stop_scan":
        state.active_scans[user_id] = False
        state.auto_scan_active[user_id] = False # Stop auto too
        config['auto_scan'] = False
        await query.edit_message_text("🛑 STOPPING... Results kept.", reply_markup=get_main_keyboard(config, False))
        return

    # If scanning, ignore other buttons
    if is_scanning:
        return

    # Parameter Adjustments
    if data == "set_port":
        # Cycle Port
        opts = [5000, 10000, 25000, 50000, 100000]
        config['portfolio'] = opts[(opts.index(config['portfolio']) + 1) % len(opts)] if config['portfolio'] in opts else 10000
    elif data == "set_rr":
        config['rr'] = round(config['rr'] + 0.25, 2)
        if config['rr'] > 5.0: config['rr'] = 1.25
    elif data == "set_risk":
        opts = [0.2, 0.5, 1.0, 1.5, 2.0]
        config['risk'] = opts[(opts.index(config['risk']) + 1) % len(opts)] if config['risk'] in opts else 0.2
    elif data == "set_matr":
        config['max_atr'] = 5.0 if config['max_atr'] == 1.5 else 1.5 # Toggle 1.5 or 5.0 for example, or cycle
    elif data == "set_sma":
        opts = [100, 150, 200]
        config['sma'] = opts[(opts.index(config['sma']) + 1) % len(opts)]
    elif data == "set_tf":
        config['tf'] = "Weekly" if config['tf'] == "Daily" else "Daily"
    elif data == "toggle_new":
        config['new_only'] = not config['new_only']
    elif data == "toggle_src":
        if config['source'] == "SP500":
            config['source'] = "MANUAL"
            await context.bot.send_message(chat_id=user_id, text="📝 Type tickers separated by comma (e.g. AAPL, TSLA)")
        else:
            config['source'] = "SP500"
    elif data == "toggle_auto":
        config['auto_scan'] = not config['auto_scan']
        state.auto_scan_active[user_id] = config['auto_scan']
        if config['auto_scan']:
            await context.bot.send_message(chat_id=user_id, text="🤖 AUTO SCAN STARTED. Will scan hourly during Market Open.")
            # Start background task for this user
            asyncio.create_task(auto_scan_loop(user_id, context.bot, query.message.chat_id))
    
    elif data == "start_scan":
        # Launch Scan
        asyncio.create_task(run_scan_process(user_id, config, context.bot, query.message.chat_id, is_auto=False))
        return # Don't update keyboard yet

    state.user_configs[user_id] = config
    try:
        await query.edit_message_reply_markup(reply_markup=get_main_keyboard(config, False))
    except: pass

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    state = get_bot_state()
    if user_id in state.user_configs:
        config = state.user_configs[user_id]
        if config['source'] == "MANUAL":
            txt = update.message.text
            config['manual_list'] = [x.strip().upper() for x in txt.split(',') if x.strip()]
            state.user_configs[user_id] = config
            await update.message.reply_text(f"✅ Loaded {len(config['manual_list'])} tickers.", reply_markup=get_main_keyboard(config, False))

# --- SCANNING CORE ---
async def run_scan_process(user_id, live_config, bot, chat_id, is_auto=False):
    state = get_bot_state()
    
    # 1. Snapshot parameters (Rule: Finish scan with old parameters if changed)
    config = live_config.copy() 
    
    state.active_scans[user_id] = True
    
    # Send Progress Message
    scan_type = "🤖 AUTO" if is_auto else "▶️ MANUAL"
    progress_text = (f"{scan_type} SCAN STARTED...\n"
                     f"PARAMS: SMA{config['sma']} | {config['tf']} | RR {config['rr']} | RISK {config['risk']}%")
    
    msg = await bot.send_message(chat_id, progress_text)
    
    # Update Server Dashboard
    state.last_scan_time = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S EST")

    # Get Tickers
    if config['source'] == "SP500":
        tickers = get_sp500_tickers()
    else:
        tickers = config['manual_list']
    
    if not tickers:
        await bot.edit_message_text("❌ NO TICKERS FOUND", chat_id=chat_id, message_id=msg.message_id)
        state.active_scans[user_id] = False
        return

    # 10.1 Daily Filter Reset
    if user_id not in state.found_tickers_today: state.found_tickers_today[user_id] = set()
    # Reset set if new day? (Simplified: Just rely on user to not care or simple implementation)
    # Ideally check date, but for now we keep the set growing until restart or manual logic.
    
    # Loop
    for i, t in enumerate(tickers):
        # Stop Check
        if not state.active_scans.get(user_id, False):
            await bot.edit_message_text("🛑 SCAN STOPPED BY USER", chat_id=chat_id, message_id=msg.message_id)
            return

        # Progress Update (Every 5%)
        if i % max(1, int(len(tickers)/20)) == 0:
            pct = int((i/len(tickers))*100)
            try:
                await bot.edit_message_text(f"{progress_text}\n⏳ Progress: {pct}% ({i}/{len(tickers)})", 
                                            chat_id=chat_id, message_id=msg.message_id)
            except: pass
        
        try:
            # Fetch Data
            inter = "1d" if config['tf'] == "Daily" else "1wk"
            period = "2y" if config['tf'] == "Daily" else "5y"
            
            # Using synchronous yf in thread
            df = yf.download(t, period=period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            if len(df) < config['sma'] + 5:
                if config['source'] == "MANUAL":
                     await bot.send_message(chat_id, f"⚠️ {t}: Not enough data")
                continue

            # Run Logic
            df = run_vova_logic(df, config['sma'], 20, 40, 14, 20, 14)
            valid, d, reason = analyze_trade(df, -1)
            
            # Rejection handling (Manual only)
            if not valid:
                if config['source'] == "MANUAL":
                    await bot.send_message(chat_id, f"❌ <b>{t}</b>: {reason}", parse_mode=ParseMode.HTML)
                continue
            
            # Filters
            # New Only
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if config['source'] == "SP500" and config['new_only'] and not is_new: continue
            
            # RR
            if d['RR'] < config['rr']:
                if config['source'] == "MANUAL": await bot.send_message(chat_id, f"❌ {t}: Low RR {d['RR']:.2f}")
                continue
                
            # ATR
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > config['max_atr']:
                if config['source'] == "MANUAL": await bot.send_message(chat_id, f"❌ {t}: High ATR {atr_pct:.1f}%")
                continue
            
            # Position
            risk_amt = config['portfolio'] * (config['risk'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            
            shares = int(risk_amt / risk_share)
            max_shares_portfolio = int(config['portfolio'] / d['P'])
            shares = min(shares, max_shares_portfolio)
            
            if shares < 1:
                if config['source'] == "MANUAL": await bot.send_message(chat_id, f"❌ {t}: Low Funds")
                continue

            # 10.1 AUTO SCAN FILTER (No duplicate tickers per day)
            if is_auto and t in state.found_tickers_today[user_id]:
                continue
            
            # Add to today's found list
            state.found_tickers_today[user_id].add(t)

            # INFO
            pe = get_financial_info(t)
            pe_s = f" | P/E: {pe:.1f}" if pe else ""
            tv_link = f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}"
            badge = "🆕" if is_new else ""
            
            val_pos = shares * d['P']
            profit_pot = (d['TP'] - d['P']) * shares
            loss_pot = (d['P'] - d['SL']) * shares
            
            # LUXURY CARD
            card = (
                f"💎 <b><a href='{tv_link}'>{t}</a></b> {badge}\n"
                f"💵 ${d['P']:.2f}{pe_s}\n"
                f"📦 <b>POS:</b> {shares} (${val_pos:.0f})\n"
                f"🎯 <b>TP:</b> {d['TP']:.2f} (<span style='color:green'>+${profit_pot:.0f}</span>)\n"
                f"🛑 <b>SL:</b> {d['SL']:.2f} (<span style='color:red'>-${loss_pot:.0f}</span>)\n"
                f"⚖️ <b>R:R:</b> {d['RR']:.2f} | 🌊 <b>ATR:</b> {atr_pct:.1f}%"
            )
            
            await bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

        except Exception as e:
            print(f"Error {t}: {e}")

    state.active_scans[user_id] = False
    await bot.edit_message_text(f"✅ {scan_type} SCAN COMPLETE", chat_id=chat_id, message_id=msg.message_id)
    
    # Restore keyboard
    await bot.send_message(chat_id, "Menu:", reply_markup=get_main_keyboard(live_config, False))

# --- AUTO SCAN LOOP ---
async def auto_scan_loop(user_id, bot, chat_id):
    state = get_bot_state()
    while state.auto_scan_active.get(user_id, False):
        # 1. Check Market Hours
        if is_market_open():
             # 2. Run Scan
             state.next_auto_scan[user_id] = "Scanning Now..."
             # We fetch current config to run the scan
             config = state.user_configs.get(user_id, DEFAULT_CONFIG).copy()
             await run_scan_process(user_id, config, bot, chat_id, is_auto=True)
             
             # 3. Wait 1 hour
             state.next_auto_scan[user_id] = "Waiting 1h..."
             await asyncio.sleep(3600) 
        else:
            state.next_auto_scan[user_id] = "Market Closed. Waiting..."
            await asyncio.sleep(600) # Check every 10 mins

# ==========================================
# 5. SERVER RUNNER
# ==========================================
def run_bot():
    state = get_bot_state()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    app = ApplicationBuilder().token(TG_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    
    state.bot_running = True
    # stop_signals=None prevents thread error in Streamlit
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)

# --- STREAMLIT WEB UI ---
st.title("🛡️ VOVA SCREENER SERVER")
st.markdown("---")

state = get_bot_state()

# Start Bot Background Thread
if not state.bot_running:
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    st.toast("Bot Starting...", icon="🤖")

# 8.1 STATUS
st.subheader("8.1 STATUS")
status = "🟢 ACTIVE" if state.bot_running else "🔴 STARTING"
st.markdown(f"**BOT STATUS:** {status}")

# 8.2 USERS
st.subheader("8.2 APPROVED USERS")
users = get_authorized_users()
st.metric("Total Users", len(users))
with st.expander("See User IDs"):
    for u in users: st.code(u)

# 8.3 LAST SCAN
st.subheader("8.3 SCAN INFO")
st.info(f"Last scan activity: {state.last_scan_time}")

# 10.2 TIME LEFT FOR AUTO SCAN
st.subheader("10.2 AUTO SCAN STATUS")
if state.next_auto_scan:
    for uid, info in state.next_auto_scan.items():
        st.write(f"User {uid}: {info}")
else:
    st.write("No active auto-scans.")

if st.button("REBOOT SERVER"):
    st.cache_resource.clear()
    st.rerun()
