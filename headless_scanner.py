import logging
import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import pytz
from datetime import datetime, time
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, JobQueue

# ==========================================
# 0. CONFIGURATION & SECRETS
# ==========================================
TG_TOKEN = "8407386703:AAFtzeEQlc0H2Ev_cccHJGE3mTdxA2c_JkA"
ADMIN_ID = "1335722880"
GITHUB_USERS_URL = "https://raw.githubusercontent.com/vladimirsk1234/vova-screener/refs/heads/main/users.txt"

# Logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. EXACT LOGIC & MATH (FROM WEB APP)
# ==========================================

# --- HELPERS ---
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Error S&P500: {e}")
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
# 2. BOT STATE & HELPERS
# ==========================================

# CONSTANTS
EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14

# User State Storage
# Structure: { chat_id: { 'config': {}, 'running': False, 'manual_scan_done': False, 'custom_scan_input': None } }
users_db = {}
# History for Auto Scan (prevent duplicates)
# Structure: { 'YYYY-MM-DD': { 'chat_id_TICKER': True } }
scan_history = {}

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
            'manual_scan_done': False,
            'auto_active': False
        }
    return users_db[chat_id]

async def check_auth(user):
    if str(user.id) == ADMIN_ID: return True
    try:
        resp = requests.get(GITHUB_USERS_URL)
        if resp.status_code == 200:
            allowed = [u.strip() for u in resp.text.splitlines() if u.strip()]
            return user.username in allowed
    except:
        return False
    return False

def is_market_open():
    """Checks if US market is open (9:30 - 16:00 ET, Mon-Fri)."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    if now.weekday() > 4: return False # Sat/Sun
    market_start = time(9, 30)
    market_end = time(16, 0)
    return market_start <= now.time() <= market_end

def format_telegram_card(t, d, shares, is_new, pe_val):
    """
    Creates a Compact, Luxury-style card using Telegram HTML.
    No CSS possible, so we use Emojis and Formatting.
    """
    tv_link = f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}"
    
    # 1. Signal & Header
    new_icon = "🆕" if is_new else ""
    header = f"💎 <a href='{tv_link}'><b>{t}</b></a> {new_icon}"
    
    # 2. Financials
    pe_str = f"P/E {pe_val:.0f}" if pe_val else "N/A"
    price_line = f"💵 <b>${d['P']:.2f}</b> | {pe_str}"
    
    # 3. Position & Target
    val_pos = shares * d['P']
    profit_pot = (d['TP'] - d['P']) * shares
    loss_pot = (d['P'] - d['SL']) * shares
    
    # Using code block for alignment simulation or just emojis
    rr_str = f"{d['RR']:.2f}"
    
    # Compact Lines
    line_pos = f"💼 <b>Pos:</b> {shares} (${val_pos:.0f})"
    line_tp = f"🎯 <b>TP:</b> {d['TP']:.2f} (+${profit_pot:.0f})"
    line_sl = f"🛑 <b>SL:</b> {d['SL']:.2f} (-${loss_pot:.0f})"
    line_risk = f"⚖️ <b>R:R:</b> {rr_str} | <b>ATR:</b> {d['ATR']:.2f}"
    
    card = (
        f"{header}\n"
        f"{price_line}\n"
        f"──────────────\n"
        f"{line_pos}\n"
        f"{line_tp}\n"
        f"{line_sl}\n"
        f"{line_risk}"
    )
    return card

# ==========================================
# 3. BOT FUNCTIONS & HANDLERS
# ==========================================

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_state = get_user_state(chat_id)
    auto_status = "🟢 ON" if user_state['auto_active'] else "🔴 OFF"
    
    keyboard = [
        [KeyboardButton("▶ START SCAN"), KeyboardButton("⏹ STOP SCAN")],
        [KeyboardButton("⚙️ SETTINGS"), KeyboardButton(f"🔄 AUTO SCAN: {auto_status}")],
        [KeyboardButton("🔎 SCAN TICKER(S)")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)
    await context.bot.send_message(chat_id=chat_id, text="📟 <b>CONTROL PANEL</b>", reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not await check_auth(user):
        await update.message.reply_text("⛔ Access Denied.")
        return
    await update.message.reply_text(f"👋 Welcome {user.first_name} to Vova Screener Bot.")
    await show_menu(update, context)

# --- SETTINGS LOGIC ---
async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    s = get_user_state(chat_id)['config']
    
    msg = (
        f"⚙️ <b>CURRENT SETTINGS</b>\n\n"
        f"💰 Portfolio: <b>${s['portfolio']}</b>\n"
        f"⚖️ Min RR: <b>{s['min_rr']}</b>\n"
        f"⚠️ Risk: <b>{s['risk_pct']}%</b>\n"
        f"📊 Max ATR: <b>{s['max_atr']}%</b>\n"
        f"📈 SMA: <b>{s['sma']}</b>\n"
        f"📅 TF: <b>{s['tf']}</b>\n"
        f"🆕 New Only: <b>{s['new_only']}</b>\n\n"
        "<i>To change, type commands:</i>\n"
        "/set_port 15000\n/set_rr 1.5\n/set_risk 0.5\n/set_atr 4\n/set_sma 150\n/set_tf Weekly\n/toggle_new"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

# Generic setter
async def generic_setter(update, context, key, type_func):
    chat_id = update.effective_chat.id
    try:
        val = type_func(context.args[0])
        get_user_state(chat_id)['config'][key] = val
        await update.message.reply_text(f"✅ {key} updated to {val}")
    except:
        await update.message.reply_text("❌ Invalid format.")

async def cmd_set_port(u, c): await generic_setter(u, c, 'portfolio', int)
async def cmd_set_rr(u, c): await generic_setter(u, c, 'min_rr', float)
async def cmd_set_risk(u, c): await generic_setter(u, c, 'risk_pct', float)
async def cmd_set_atr(u, c): await generic_setter(u, c, 'max_atr', float)
async def cmd_set_sma(u, c): 
    val = int(c.args[0])
    if val in [100, 150, 200]:
        get_user_state(u.effective_chat.id)['config']['sma'] = val
        await u.message.reply_text(f"✅ SMA updated to {val}")
    else: await u.message.reply_text("❌ Use 100, 150 or 200")
async def cmd_set_tf(u, c):
    val = c.args[0].capitalize()
    if val in ['Daily', 'Weekly']:
        get_user_state(u.effective_chat.id)['config']['tf'] = val
        await u.message.reply_text(f"✅ Timeframe updated to {val}")
    else: await u.message.reply_text("❌ Use Daily or Weekly")
async def cmd_toggle_new(u, c):
    s = get_user_state(u.effective_chat.id)['config']
    s['new_only'] = not s['new_only']
    await u.message.reply_text(f"✅ New Signals Only: {s['new_only']}")

# --- SCANNING LOGIC ---
async def run_scan_process(context: ContextTypes.DEFAULT_TYPE, chat_id, tickers, mode="Manual"):
    state = get_user_state(chat_id)
    
    # 1. Snapshot Parameters (Freeze)
    p = state['config'].copy() 
    
    # Progress UI
    progress_msg = await context.bot.send_message(chat_id, f"🚀 <b>Starting {mode} Scan...</b>\nTarget: {len(tickers)} tickers", parse_mode=ParseMode.HTML)
    
    total = len(tickers)
    found = 0
    today_str = datetime.now().strftime('%Y-%m-%d')
    if today_str not in scan_history: scan_history[today_str] = {}

    for i, t in enumerate(tickers):
        # Check Stop Flag (Only for manual loops, auto usually runs fully)
        if not state['running'] and mode == "Manual":
            await context.bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id, text="⏹ <b>Scan Stopped by User.</b>", parse_mode=ParseMode.HTML)
            return

        # Update Progress Bar every 5% or 10 tickers to avoid rate limits
        if i % 10 == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            bar = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
            try:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id, text=f"🔍 <b>Scanning...</b> {pct}%\n{bar}\nChecking: {t}", parse_mode=ParseMode.HTML)
            except: pass

        # Skip duplicate for Auto Mode
        if mode == "Auto":
            uniq_key = f"{chat_id}_{t}"
            if uniq_key in scan_history[today_str]: continue

        try:
            # Data Fetch
            inter = "1d" if p['tf'] == "Daily" else "1wk"
            per = "2y" if p['tf'] == "Daily" else "5y"
            
            # Using asyncio.to_thread for blocking yfinance
            df = await asyncio.to_thread(yf.download, t, period=per, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            # Logic Checks
            reason = "OK"
            valid = True
            
            if len(df) < p['sma'] + 5: 
                valid = False; reason = "NO DATA"
            else:
                df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
                valid, d, reason = analyze_trade(df, -1)
            
            # Manual Mode Specific: Report rejection if requested (for single ticker lists)
            if mode == "Single" and not valid:
                await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: {reason}", parse_mode=ParseMode.HTML)
                continue

            if not valid: continue

            # Filters
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if p['new_only'] and not is_new: continue

            if d['RR'] < p['min_rr']:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: Low RR ({d['RR']:.2f})", parse_mode=ParseMode.HTML)
                continue
            
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > p['max_atr']:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: High ATR ({atr_pct:.1f}%)", parse_mode=ParseMode.HTML)
                continue
                
            # Position Sizing
            risk_amt = p['portfolio'] * (p['risk_pct'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            
            shares = int(risk_amt / risk_share)
            max_shares = int(p['portfolio'] / d['P'])
            shares = min(shares, max_shares)
            
            if shares < 1:
                if mode == "Single": await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: Low Funds", parse_mode=ParseMode.HTML)
                continue

            # Success - Send Card
            pe = await asyncio.to_thread(get_financial_info, t)
            card = format_telegram_card(t, d, shares, is_new, pe)
            
            # SEND IMMEDIATELY
            await context.bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            found += 1
            
            # Record history for auto
            if mode == "Auto":
                scan_history[today_str][f"{chat_id}_{t}"] = True

        except Exception as e:
            continue

    # Finish
    state['running'] = False
    if mode == "Manual": state['manual_scan_done'] = True
    
    final_text = f"🏁 <b>Scan Complete.</b> Found: {found}"
    await context.bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id, text=final_text, parse_mode=ParseMode.HTML)

# --- COMMAND HANDLERS ---

async def handle_start_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_user_state(chat_id)
    
    if state['running']:
        await update.message.reply_text("⚠️ Scan already running.")
        return
        
    state['running'] = True
    tickers = get_sp500_tickers()
    if not tickers:
        await update.message.reply_text("❌ Error fetching S&P 500 list.")
        state['running'] = False
        return

    # Trigger async scan
    asyncio.create_task(run_scan_process(context, chat_id, tickers, mode="Manual"))

async def handle_stop_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_user_state(chat_id)
    if state['running']:
        state['running'] = False
        await update.message.reply_text("🛑 Stopping scan... please wait for current ticker.")
    else:
        await update.message.reply_text("💤 No scan running.")

async def handle_single_scan_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⌨️ Type tickers separated by comma (e.g., AAPL, TSLA, NVDA):")
    get_user_state(update.effective_chat.id)['expecting_input'] = True

async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text
    state = get_user_state(chat_id)

    # Menu Buttons Logic
    if text == "▶ START SCAN": await handle_start_scan(update, context); return
    if text == "⏹ STOP SCAN": await handle_stop_scan(update, context); return
    if text == "⚙️ SETTINGS": await settings_menu(update, context); return
    if text.startswith("🔄 AUTO SCAN"): await toggle_auto_scan(update, context); return
    if text == "🔎 SCAN TICKER(S)": await handle_single_scan_request(update, context); return

    # Ticker Input Logic
    if state.get('expecting_input'):
        state['expecting_input'] = False
        tickers = [t.strip().upper() for t in text.split(',') if t.strip()]
        state['running'] = True
        asyncio.create_task(run_scan_process(context, chat_id, tickers, mode="Single"))
        return

# --- AUTO SCAN SYSTEM ---

async def hourly_auto_job(context: ContextTypes.DEFAULT_TYPE):
    # This runs every hour. We iterate over all users with auto_active = True
    for chat_id, data in users_db.items():
        if data.get('auto_active', False):
            # Check market open
            if is_market_open():
                # Check if manual scan was done at least once
                if data.get('manual_scan_done', False):
                     tickers = get_sp500_tickers()
                     if tickers:
                         # Run quietly without big progress bars or stop checks
                         asyncio.create_task(run_scan_process(context, chat_id, tickers, mode="Auto"))

async def toggle_auto_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_user_state(chat_id)
    
    if not state['manual_scan_done']:
        await update.message.reply_text("⚠️ You must run a Manual Scan at least once before enabling Auto.")
        return

    state['auto_active'] = not state['auto_active']
    await show_menu(update, context) # Refresh button status

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    # Build App
    application = Application.builder().token(TG_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    
    # Settings Commands
    application.add_handler(CommandHandler("set_port", cmd_set_port))
    application.add_handler(CommandHandler("set_rr", cmd_set_rr))
    application.add_handler(CommandHandler("set_risk", cmd_set_risk))
    application.add_handler(CommandHandler("set_atr", cmd_set_atr))
    application.add_handler(CommandHandler("set_sma", cmd_set_sma))
    application.add_handler(CommandHandler("set_tf", cmd_set_tf))
    application.add_handler(CommandHandler("toggle_new", cmd_toggle_new))
    
    # Text Handler (Menu & Inputs)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    
    # Job Queue for Auto Scan
    job_queue = application.job_queue
    job_queue.run_repeating(hourly_auto_job, interval=3600, first=10)

    # --- STREAMLIT FIX FOR RUNTIME ERROR ---
    print("💎 Vova Bot is Running...")
    # This specifically fixes the "add_signal_handler" error on Streamlit Cloud
    application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None, close_loop=False)

if __name__ == "__main__":
    main()
