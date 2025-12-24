import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import asyncio
import threading
import pytz
import nest_asyncio
from datetime import datetime, time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters

# Применяем патч для работы asyncio в среде Streamlit
nest_asyncio.apply()

# ==========================================
# 0. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================
try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets["GITHUB_USERS_URL"]
except Exception as e:
    st.error(f"Ошибка секретов: {e}. Убедитесь, что TG_TOKEN и ADMIN_ID заданы.")
    st.stop()

# Глобальное состояние для связи Бота и Веб-интерфейса
if 'BOT_STATS' not in st.session_state:
    st.session_state.BOT_STATS = {
        "status": "STOPPED",
        "last_scan": "Никогда",
        "approved_users": 1, # Админ всегда одобрен
        "auto_scan_active": False,
        "next_auto_scan": "Нет"
    }

# ==========================================
# 1. ЛОГИКА VOVA SCREENER (100% COPY)
# ==========================================
# --- HELPER FUNCTIONS ---
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
    
    # State Variables (simulate 'var' in Pine)
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]; s_l = l_a[0]
    
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        
        # Access "Previous" values (index [1] in Pine)
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h
        prev_sl = s_l
        
        # Break Detection
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1: # Bearish Break (Up -> Down)
                # Did we make a HH before breaking?
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                
                # Update Memory
                pk_hh = is_hh
                last_pk = prev_sh # LAST CONFIRMED PEAK (TARGET)
                
                # Reset State
                s_state = -1
                s_h = h; s_l = l
                s_crit = h # Initial stop for downtrend
                
            else: # Bullish Break (Down -> Up)
                # Did we make a HL before breaking?
                is_hl = True if np.isnan(last_tr) else (prev_sl > last_tr)
                
                # Update Memory
                tr_hl = is_hl
                last_tr = prev_sl
                
                # Reset State
                s_state = 1
                s_h = h; s_l = l
                s_crit = l # Initial stop for uptrend
        else:
            # Continue State
            s_state = prev_st
            
            if s_state == 1: # Uptrend
                if h >= s_h: s_h = h
                
                # Trailing Logic: if high >= previous seqHigh, trail stop to low
                if h >= prev_sh: s_crit = l
                else: s_crit = prev_cr
                
            elif s_state == -1: # Downtrend
                if l <= s_l: s_l = l
                
                # Trailing Logic
                if l <= prev_sl: s_crit = h
                else: s_crit = prev_cr
                
            else: # Init state 0
                if c > prev_sh: 
                    s_state = 1; s_crit = l
                elif c < prev_sl: 
                    s_state = -1; s_crit = h
                else:
                    s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        
        # Store results for this bar
        seq_st[i] = s_state
        crit_lvl[i] = s_crit
        res_peak[i] = last_pk # TP IS LAST CONFIRMED PEAK
        res_struct[i] = (pk_hh and tr_hl)

    # --- Super Trend Logic (Vectorized) ---
    adx_str = adx >= adx_thr
    
    # Bullish: ADX+DI, Elder Impulse (EMA+Hist Rising), EFI > 0
    bull = (adx_str & (p_di > m_di)) & \
           ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & \
           (efi > 0)
           
    # Bearish: ADX-DI, Elder Impulse (EMA+Hist Falling), EFI < 0
    bear = (adx_str & (m_di > p_di)) & \
           ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & \
           (efi < 0)
           
    t_st = np.zeros(n, dtype=int)
    t_st[bull] = 1
    t_st[bear] = -1
    
    # Assign to DF
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
    
    # 1. Validation Rules
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(r['SMA']) or r['Close'] <= r['SMA']: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(r['Peak']) or np.isnan(r['Crit']): errs.append("NO DATA")
    
    if errs: return False, {}, " ".join(errs)
    
    # 2. Key Levels
    price = r['Close']
    tp = r['Peak'] # LAST CONFIRMED PEAK (HH)
    crit = r['Crit']
    atr = r['ATR']
    
    # 3. Safer SL Selection
    # For a LONG trade, we want the stop that gives the trade more room (is lower).
    # SL_Struct = Critical Level
    # SL_ATR = Price - 1.5 ATR
    sl_struct = crit
    sl_atr = price - atr
    
    # Pick minimum (lowest price)
    final_sl = min(sl_struct, sl_atr)
    
    # 4. Geometry Check
    risk = price - final_sl
    reward = tp - price
    
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    # 5. Calculate Real Monetary RR
    rr = reward / risk
    
    return True, {
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": rr, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# ==========================================
# 2. TELEGRAM BOT LOGIC
# ==========================================
# --- Bot State & Globals ---
bot_running = False
scan_active = False
found_today = set() # Чтобы не показывать тикер дважды

# Параметры по умолчанию
DEFAULT_PARAMS = {
    'portfolio': 10000,
    'min_rr': 1.25,
    'risk_pct': 0.2,
    'max_atr': 5.0,
    'sma_p': 200,
    'tf': 'Daily',
    'new_only': True,
    'auto_scan': False
}

user_params = DEFAULT_PARAMS.copy()

# --- Functions ---
def get_lux_card(t, d, shares, val_pos, profit_pot, loss_pot, pe, is_new):
    # COOL LOOKING LUXURY CARDS
    # 1 LINE PER INFO, MAX TEXT
    tv_link = f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}"
    badge = "🆕" if is_new else ""
    pe_str = f"(P/E: {pe:.1f})" if pe else ""
    atr_pct = (d['ATR']/d['P'])*100
    
    # HTML FORMATTING
    msg = (
        f"{badge} 💎 <b><a href='{tv_link}'>{t}</a></b> {pe_str}\n"
        f"💵 <b>Price:</b> ${d['P']:.2f}\n"
        f"⚖️ <b>Pos:</b> {shares} (~${val_pos:.0f})\n"
        f"🎯 <b>TP:</b> {d['TP']:.2f} (<i>+${profit_pot:.0f}</i>)\n"
        f"🛑 <b>SL:</b> {d['SL']:.2f} (<i>-${loss_pot:.0f}</i>) [{d['SL_Type']}]\n"
        f"📉 <b>Crit:</b> {d['Crit']:.2f}\n"
        f"📊 <b>ATR:</b> ${d['ATR']:.2f} ({atr_pct:.1f}%)"
    )
    return msg

async def perform_scan(update: Update, context: ContextTypes.DEFAULT_TYPE, manual_tickers=None):
    global scan_active, user_params, found_today
    scan_active = True
    
    chat_id = update.effective_chat.id if update else ADMIN_ID
    
    # Захватываем параметры на момент старта
    p = user_params.copy()
    
    # Тикеры
    if manual_tickers:
        tickers = [x.strip().upper() for x in manual_tickers.split(',') if x.strip()]
        src_type = "MANUAL"
    else:
        tickers = get_sp500_tickers()
        src_type = "S&P 500"
        
    status_msg = await context.bot.send_message(
        chat_id=chat_id, 
        text=f"🚀 <b>SCAN STARTED</b>\nTYPE: {src_type}\nPARAMS: RR>{p['min_rr']} | Risk {p['risk_pct']}% | SMA {p['sma_p']}",
        parse_mode=ParseMode.HTML
    )

    st.session_state.BOT_STATS["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14
    
    total = len(tickers)
    found_count = 0
    
    for i, t in enumerate(tickers):
        # Check Stop
        if not scan_active and not manual_tickers: # Manual always finishes
            await context.bot.edit_message_text(f"🛑 <b>SCAN STOPPED BY USER</b>", chat_id=chat_id, message_id=status_msg.message_id, parse_mode=ParseMode.HTML)
            return

        # Progress Update (every 10 or 10%)
        if i % 10 == 0 or i == total - 1:
            pct = int((i/total)*100)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, 
                    message_id=status_msg.message_id, 
                    text=f"⏳ <b>SCANNING...</b> {pct}%\nFound: {found_count}\nTicker: {t}",
                    parse_mode=ParseMode.HTML
                )
            except: pass

        try:
            inter = "1d" if p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if p['tf'] == "Daily" else "5y"
            df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            # --- Validations (Logic Copy) ---
            if len(df) < p['sma_p'] + 5:
                if manual_tickers: await context.bot.send_message(chat_id, f"❌ {t}: NO DATA")
                continue

            df = run_vova_logic(df, p['sma_p'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            valid, d, reason = analyze_trade(df, -1)
            
            if not valid:
                if manual_tickers: await context.bot.send_message(chat_id, f"❌ {t}: {reason}")
                continue
            
            # Filters
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if p['new_only'] and not is_new and not manual_tickers: continue
            if d['RR'] < p['min_rr']:
                if manual_tickers: await context.bot.send_message(chat_id, f"❌ {t}: LOW RR ({d['RR']:.2f})")
                continue
                
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > p['max_atr']:
                if manual_tickers: await context.bot.send_message(chat_id, f"❌ {t}: HIGH VOL ({atr_pct:.1f}%)")
                continue
            
            # Money Mgmt
            risk_amt = p['portfolio'] * (p['risk_pct'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue 
            
            shares = int(risk_amt / risk_share)
            max_shares_portfolio = int(p['portfolio'] / d['P'])
            shares = min(shares, max_shares_portfolio)
            
            if shares < 1:
                if manual_tickers: await context.bot.send_message(chat_id, f"❌ {t}: LOW FUNDS")
                continue

            # DUPLICATE CHECK
            date_key = datetime.now().strftime("%Y-%m-%d") + t
            if date_key in found_today and not manual_tickers:
                continue
            found_today.add(date_key)

            # --- SUCCESS ---
            pe = get_financial_info(t)
            val_pos = shares * d['P']
            profit_pot = (d['TP'] - d['P']) * shares
            loss_pot = (d['P'] - d['SL']) * shares
            
            card = get_lux_card(t, d, shares, val_pos, profit_pot, loss_pot, pe, is_new)
            
            await context.bot.send_message(chat_id=chat_id, text=card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            found_count += 1
            
        except Exception as e:
            print(f"Err {t}: {e}")
            continue

    scan_active = False
    await context.bot.send_message(chat_id=chat_id, text=f"✅ <b>SCAN COMPLETE</b>. Found: {found_count}", parse_mode=ParseMode.HTML)

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    await show_menu(update, context)

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = user_params
    auto_status = "🟢 ON" if p['auto_scan'] else "🔴 OFF"
    
    keyboard = [
        [InlineKeyboardButton("▶ START SCAN", callback_data="start_scan"), InlineKeyboardButton("⏹ STOP", callback_data="stop_scan")],
        [InlineKeyboardButton(f"🔄 AUTO: {auto_status}", callback_data="toggle_auto")],
        [InlineKeyboardButton("MANUAL TICKER INPUT", callback_data="manual_input")],
        [InlineKeyboardButton(f"💰 PORT: ${p['portfolio']}", callback_data="set_port"), InlineKeyboardButton(f"⚖️ RR: {p['min_rr']}", callback_data="set_rr")],
        [InlineKeyboardButton(f"⚠️ RISK: {p['risk_pct']}%", callback_data="set_risk"), InlineKeyboardButton(f"📊 ATR: {p['max_atr']}%", callback_data="set_atr")],
        [InlineKeyboardButton(f"📈 SMA: {p['sma_p']}", callback_data="toggle_sma"), InlineKeyboardButton(f"📅 TF: {p['tf']}", callback_data="toggle_tf")],
        [InlineKeyboardButton(f"🆕 NEW ONLY: {p['new_only']}", callback_data="toggle_new")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    txt = "💎 <b>VOVA SCREENER BOT</b> 💎\nUse buttons below to control."
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text=txt, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(txt, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    global user_params, scan_active
    
    if data == "start_scan":
        asyncio.create_task(perform_scan(update, context))
    
    elif data == "stop_scan":
        scan_active = False
        await context.bot.send_message(chat_id=ADMIN_ID, text="🛑 STOPPING...")
    
    elif data == "toggle_auto":
        user_params['auto_scan'] = not user_params['auto_scan']
        st.session_state.BOT_STATS["auto_scan_active"] = user_params['auto_scan']
        
        # Job Logic
        job_q = context.job_queue
        current_jobs = job_q.get_jobs_by_name("auto_scan_job")
        
        if user_params['auto_scan']:
            if not current_jobs:
                job_q.run_repeating(auto_scan_job, interval=3600, first=10, name="auto_scan_job")
                await context.bot.send_message(ADMIN_ID, "⏰ <b>AUTO SCAN: STARTED (Hourly)</b>", parse_mode=ParseMode.HTML)
        else:
            for job in current_jobs: job.schedule_removal()
            await context.bot.send_message(ADMIN_ID, "⏰ <b>AUTO SCAN: STOPPED</b>", parse_mode=ParseMode.HTML)
        
        await show_menu(update, context)

    # Param Toggles
    elif data == "toggle_sma":
        opts = [100, 150, 200]
        curr = opts.index(user_params['sma_p'])
        user_params['sma_p'] = opts[(curr + 1) % 3]
        await show_menu(update, context)
        
    elif data == "toggle_tf":
        user_params['tf'] = "Weekly" if user_params['tf'] == "Daily" else "Daily"
        await show_menu(update, context)
        
    elif data == "toggle_new":
        user_params['new_only'] = not user_params['new_only']
        await show_menu(update, context)
        
    elif data == "manual_input":
        await context.bot.send_message(ADMIN_ID, "⌨️ <b>TYPE TICKERS</b> separated by comma (e.g. AAPL, TSLA, NVDA):", parse_mode=ParseMode.HTML)
        context.user_data['awaiting_input'] = 'manual'
        
    elif data.startswith("set_"):
        param = data.split("_")[1]
        context.user_data['awaiting_input'] = param
        await context.bot.send_message(ADMIN_ID, f"⌨️ Type new value for <b>{param.upper()}</b>:", parse_mode=ParseMode.HTML)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    state = context.user_data.get('awaiting_input')
    
    if not state: return
    
    text = update.message.text
    
    if state == 'manual':
        asyncio.create_task(perform_scan(update, context, manual_tickers=text))
        context.user_data['awaiting_input'] = None
        return

    try:
        val = float(text)
        if state == 'port': user_params['portfolio'] = int(val)
        elif state == 'rr': user_params['min_rr'] = max(1.25, val)
        elif state == 'risk': user_params['risk_pct'] = max(0.2, val)
        elif state == 'atr': user_params['max_atr'] = val
        
        await update.message.reply_text(f"✅ Set {state} to {val}")
        await show_menu(update, context)
        context.user_data['awaiting_input'] = None
    except:
        await update.message.reply_text("❌ Invalid number")

# --- Auto Scan Job ---
async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    # Check Market Hours (US Eastern)
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Пн-Пт, 9:30 - 16:00
    if now.weekday() < 5 and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 16:
        st.session_state.BOT_STATS["last_scan"] = now.strftime("%Y-%m-%d %H:%M:%S")
        await context.bot.send_message(ADMIN_ID, "⏰ <b>AUTO SCAN STARTING...</b>", parse_mode=ParseMode.HTML)
        # Trigger scan logic (re-using manual scan logic but without update object)
        # We need to craft a fake update or refactor perform_scan. 
        # Refactoring perform_scan to handle None update
        await perform_scan(None, context)
        
        # Time left logic for Web
        st.session_state.BOT_STATS["next_auto_scan"] = "Checking next hour..."
    else:
        st.session_state.BOT_STATS["next_auto_scan"] = "Market Closed"

# --- Bot Thread ---
def run_bot():
    global bot_running
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    app = ApplicationBuilder().token(TG_TOKEN).post_init(on_startup).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    
    bot_running = True
    st.session_state.BOT_STATS["status"] = "ACTIVE"
    app.run_polling()

async def on_startup(app):
    await app.bot.send_message(chat_id=ADMIN_ID, text="🤖 <b>VOVA SCREENER BOT ONLINE</b>", parse_mode=ParseMode.HTML)

if not bot_running:
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()

# ==========================================
# 3. STREAMLIT WEB UI
# ==========================================
st.set_page_config(page_title="Screener Vova (Terminal)", layout="wide", page_icon="💎")

# CSS STYLING
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    .block-container { padding-top: 2rem !important; }
    
    /* TERMINAL CARD */
    .ticker-card { background: #0f0f0f; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px; margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.5); min-height: 110px; display: flex; flex-direction: column; justify-content: space-between; }
    .ticker-card:hover { border-color: #00e676; }
    .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #222; padding-bottom: 4px; margin-bottom: 6px; }
    .t-link { font-size: 14px; font-weight: 800; color: #448aff !important; text-decoration: none; }
    .t-price { font-size: 13px; color: #eceff1; font-weight: 700; }
    .t-pe { font-size: 9px; color: #607d8b; margin-left: 4px; }
    .new-badge { background: #00e676; color: #000; font-size: 8px; padding: 1px 4px; border-radius: 3px; margin-left: 5px; font-weight: 900; }
    .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
    .stat-row { background: #161616; padding: 3px 5px; border-radius: 3px; border: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
    .lbl { font-size: 8px; color: #78909c; font-weight: 700; }
    .val { font-size: 11px; font-weight: 700; color: #e0e0e0; text-align: right; }
    .sub { font-size: 9px; font-weight: 500; opacity: 0.8; text-align: right; display: block; }
    .c-green { color: #00e676; } .c-red { color: #ff1744; } .c-blue { color: #448aff; } .c-gold { color: #ffab00; }
    
    /* BOT STATUS BOX */
    .bot-stat-box { background: #1e1e1e; padding: 10px; border-radius: 5px; border-left: 5px solid #448aff; margin-bottom: 20px; display: flex; gap: 20px; align-items: center; color: white; }
    .bs-item { display: flex; flex-direction: column; }
    .bs-head { font-size: 10px; color: #888; text-transform: uppercase; }
    .bs-val { font-size: 14px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- BOT STATUS DISPLAY ---
s = st.session_state.BOT_STATS
st.markdown(f"""
<div class="bot-stat-box">
    <div class="bs-item"><span class="bs-head">BOT STATUS</span><span class="bs-val" style="color: {'#00e676' if s['status']=='ACTIVE' else '#ff1744'}">{s['status']}</span></div>
    <div class="bs-item"><span class="bs-head">APPROVED USERS</span><span class="bs-val">{s['approved_users']}</span></div>
    <div class="bs-item"><span class="bs-head">AUTO SCAN</span><span class="bs-val">{str(s['auto_scan_active']).upper()}</span></div>
    <div class="bs-item"><span class="bs-head">LAST SCAN</span><span class="bs-val">{s['last_scan']}</span></div>
    <div class="bs-item"><span class="bs-head">NEXT AUTO</span><span class="bs-val">{s['next_auto_scan']}</span></div>
</div>
""", unsafe_allow_html=True)

# --- ORIGINAL WEB SCREENER UI (BELOW) ---
st.header("Terminal View")

# (Здесь мы можем просто отобразить заглушку или реальный скринер, 
# если ты хочешь использовать веб-интерфейс параллельно. 
# Код ниже - это облегченная версия твоего оригинального UI для просмотра результатов, 
# так как логика сканирования теперь в основном в боте, но можно оставить и ручной веб-скан)

# ... Вставляю твой оригинальный Sidebar и код рендеринга результатов для совместимости ...
# ... (Код ниже дублирует твой оригинальный код интерфейса для полноты) ...

if 'scanning' not in st.session_state: st.session_state.scanning = False
if 'results' not in st.session_state: st.session_state.results = []
if 'rejected' not in st.session_state: st.session_state.rejected = []

st.sidebar.header("WEB CONTROLS")
src = st.sidebar.radio("SOURCE", ["All S&P 500", "Manual Input"], key="web_src")
man_txt = ""
if src == "Manual Input": man_txt = st.sidebar.text_area("TICKERS", "AAPL, TSLA")

if st.sidebar.button("▶ START WEB SCAN"):
    st.session_state.scanning = True
    st.session_state.results = []
    
    # Simple Web Scan Loop (Simplified to avoid duplicate code block, calling Logic)
    tickers = get_sp500_tickers() if src == "All S&P 500" else [x.strip() for x in man_txt.split(',')]
    bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        bar.progress((i+1)/len(tickers))
        try:
            df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=False, multi_level_index=False)
            if len(df) < 205: continue
            
            df = run_vova_logic(df, 200, 20, 40, 14, 20, 14)
            valid, d, _ = analyze_trade(df, -1)
            
            if valid:
                # Reuse Bot Params for Web Display Logic or Defaults
                shares = int(10000 / d['P']) # Default dummy
                html = f"""
                <div class="ticker-card">
                    <div class="card-header"><div><span class="t-link">{t}</span></div><div><span class="t-price">${d['P']:.2f}</span></div></div>
                    <div class="card-grid">
                        <div class="stat-row"><span class="lbl">TP</span> <span class="val c-green">{d['TP']:.2f}</span></div>
                        <div class="stat-row"><span class="lbl">SL</span> <span class="val c-red">{d['SL']:.2f}</span></div>
                        <div class="stat-row"><span class="lbl">RR</span> <span class="val c-blue">{d['RR']:.2f}</span></div>
                    </div>
                </div>
                """
                st.session_state.results.append(html)
        except: pass
        
    st.session_state.scanning = False
    st.rerun()

# Display Results
cols = st.columns(6)
for idx, h in enumerate(st.session_state.results):
    with cols[idx % 6]:
        st.markdown(h, unsafe_allow_html=True)
