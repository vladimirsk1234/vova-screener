import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import asyncio
import threading
import time
import pytz
from datetime import datetime, time as dt_time
import nest_asyncio

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters

# Применяем патч для работы асинхронного бота внутри Streamlit
nest_asyncio.apply()

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================
st.set_page_config(page_title="Vova Bot Server", layout="wide", page_icon="🤖")

# Получаем секреты
TOKEN = st.secrets.get("TG_TOKEN", "")
ADMIN_ID = str(st.secrets.get("ADMIN_ID", ""))

# ==========================================
# 2. VOVA LOGIC (EXACT COPY 100%)
# ==========================================
# --- HELPER FUNCTIONS ---
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception:
        return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# --- INDICATOR MATH ---
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
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
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

    adx_str = adx >= adx_thr
    bull = (adx_str & (p_di > m_di)) & ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & (efi > 0)
    bear = (adx_str & (m_di > p_di)) & ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & (efi < 0)
            
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
# 3. GLOBAL BOT STATE
# ==========================================
class BotState:
    def __init__(self):
        self.users = {ADMIN_ID} # Approved users
        self.is_scanning = False
        self.stop_requested = False
        self.last_scan_time = "Never"
        self.next_auto_scan = "Not scheduled"
        self.active_scan_msg_id = None
        self.active_chat_id = None
        # Parameters
        self.portfolio_size = 10000
        self.min_rr = 1.25
        self.risk_per_trade = 0.2
        self.max_atr = 5.0
        self.sma_period = 200
        self.timeframe = "Daily"
        self.new_signals_only = True
        self.sent_signals = set() # To prevent duplicates in auto mode (Ticker + Date)

BOT_STATE = BotState()

# ==========================================
# 4. TELEGRAM BOT LOGIC
# ==========================================

# --- Keyboards ---
def get_main_menu_keyboard():
    s = BOT_STATE
    btn_tf = f"TIMEFRAME: {s.timeframe}"
    btn_sma = f"SMA: {s.sma_period}"
    btn_rr = f"RR: {s.min_rr}"
    btn_risk = f"RISK: {s.risk_per_trade}%"
    btn_atr = f"ATR MAX: {s.max_atr}%"
    btn_new = f"NEW ONLY: {'✅' if s.new_signals_only else '❌'}"
    btn_port = f"PORTFOLIO: ${s.portfolio_size}"

    keyboard = [
        [InlineKeyboardButton("▶ START SCAN", callback_data="start_scan"), 
         InlineKeyboardButton("⏹ STOP", callback_data="stop_scan")],
        [InlineKeyboardButton(btn_tf, callback_data="set_tf"),
         InlineKeyboardButton(btn_sma, callback_data="set_sma")],
        [InlineKeyboardButton(btn_rr, callback_data="set_rr"),
         InlineKeyboardButton(btn_risk, callback_data="set_risk")],
        [InlineKeyboardButton(btn_atr, callback_data="set_atr"),
         InlineKeyboardButton(btn_new, callback_data="toggle_new")],
        [InlineKeyboardButton(btn_port, callback_data="set_port")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Formatters ---
def format_luxury_card(ticker, d, shares):
    # Telegram HTML (Limited tags)
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    pe = get_financial_info(ticker)
    pe_str = f"PE:{pe:.0f}" if pe else ""
    
    val_pos = shares * d['P']
    profit_pot = (d['TP'] - d['P']) * shares
    loss_pot = (d['P'] - d['SL']) * shares
    atr_pct = (d['ATR']/d['P'])*100
    
    # Emojis for new signal
    badge = "💎 <b>NEW</b>"
    
    card = (
        f"<b>{badge}</b> | <a href='{tv_link}'>{ticker}</a> | <b>${d['P']:.2f}</b> ({pe_str})\n"
        f"⚖️ RR:<b>{d['RR']:.2f}</b> | 💰 Pos: <b>{shares}</b> (${val_pos:.0f})\n"
        f"🎯 TP: <b>{d['TP']:.2f}</b> (+${profit_pot:.0f}) | 🛑 SL: <b>{d['SL']:.2f}</b> (-${loss_pot:.0f})\n"
        f"📉 Crit: <b>{d['Crit']:.2f}</b> | 📊 ATR: <b>{atr_pct:.1f}%</b> (${d['ATR']:.2f})"
    )
    return card

# --- Logic Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id not in BOT_STATE.users and user_id != ADMIN_ID:
        await update.message.reply_text("⛔ Access Denied.")
        return
    await update.message.reply_text("💎 Vova Screener Bot v1.0\nИспользуйте меню ниже:", reply_markup=get_main_menu_keyboard())

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    s = BOT_STATE

    if data == "start_scan":
        if s.is_scanning:
            await query.edit_message_text("⚠️ Уже сканирую...", reply_markup=get_main_menu_keyboard())
            return
        # Start scanning process
        s.stop_requested = False
        s.is_scanning = True
        s.active_chat_id = query.message.chat_id
        
        # Reset previous results logic handled inside scan function
        msg = await query.message.reply_text("⏳ Запуск сканирования S&P 500...")
        s.active_scan_msg_id = msg.message_id
        
        # Run scan in background task to not block bot
        context.application.create_task(perform_scan(context, "ALL"))

    elif data == "stop_scan":
        if s.is_scanning:
            s.stop_requested = True
            await query.message.reply_text("🛑 Остановка после текущего тикера...")
        else:
            await query.message.reply_text("ℹ️ Сканирование не запущено.")

    elif data == "toggle_new":
        s.new_signals_only = not s.new_signals_only
        await query.edit_message_reply_markup(reply_markup=get_main_menu_keyboard())

    elif data == "set_sma":
        opts = [100, 150, 200]
        try:
            curr_idx = opts.index(s.sma_period)
            s.sma_period = opts[(curr_idx + 1) % len(opts)]
        except: s.sma_period = 200
        await query.edit_message_reply_markup(reply_markup=get_main_menu_keyboard())
    
    elif data == "set_tf":
        s.timeframe = "Weekly" if s.timeframe == "Daily" else "Daily"
        await query.edit_message_reply_markup(reply_markup=get_main_menu_keyboard())
        
    elif data in ["set_rr", "set_risk", "set_atr", "set_port"]:
        await query.message.reply_text("⚠️ Для изменения числовых параметров пока используйте веб-интерфейс или хардкод (упрощено для Telegram меню).")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id not in BOT_STATE.users: return
    
    text = update.message.text
    # Check if it looks like tickers
    if "," in text or text.isupper():
        tickers = [t.strip().upper() for t in text.split(',') if t.strip()]
        if tickers:
            msg = await update.message.reply_text(f"🔎 Сканирую {len(tickers)} тикеров: {', '.join(tickers)}...")
            BOT_STATE.active_chat_id = update.message.chat_id
            BOT_STATE.active_scan_msg_id = msg.message_id
            BOT_STATE.stop_requested = False
            BOT_STATE.is_scanning = True
            context.application.create_task(perform_scan(context, tickers))
    else:
        await update.message.reply_text("ℹ️ Введите тикеры через запятую (например: AAPL, TSLA) для ручного скана.")

# --- SCANNING CORE TASK ---
async def perform_scan(context, target):
    s = BOT_STATE
    try:
        # 1. Get Tickers
        if target == "ALL":
            tickers = get_sp500_tickers()
            mode_desc = "S&P 500"
        else:
            tickers = target
            mode_desc = "Manual"

        total = len(tickers)
        found_count = 0
        
        # 2. Logic Constants
        EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14
        
        # 3. Loop
        for i, t in enumerate(tickers):
            if s.stop_requested:
                break
                
            # Update Progress Bar (Edit Message) every 5% or 10 tickers
            pct = int((i / total) * 100)
            if i % 10 == 0 or i == total - 1:
                try:
                    await context.bot.edit_message_text(
                        chat_id=s.active_chat_id,
                        message_id=s.active_scan_msg_id,
                        text=f"🔄 <b>Scanning {mode_desc}</b>\nProgress: {pct}% ({i}/{total})\nParams: TF={s.timeframe}, SMA={s.sma_period}, Risk={s.risk_per_trade}%",
                        parse_mode=ParseMode.HTML
                    )
                except: pass

            try:
                # Async data fetch wrapper needed? yfinance uses threads but is sync. 
                # Blocking call is okay in a separate thread/task for now.
                inter = "1d" if s.timeframe == "Daily" else "1wk"
                period = "2y" if s.timeframe == "Daily" else "5y"
                
                # Suppress stdout for yfinance
                df = yf.download(t, period=period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                
                if len(df) < s.sma_period + 5:
                    if mode_desc == "Manual":
                        await context.bot.send_message(s.active_chat_id, f"❌ {t}: Not enough data")
                    continue

                # Run Logic
                df = run_vova_logic(df, s.sma_period, EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
                valid, d, reason = analyze_trade(df, -1)

                if not valid:
                    if mode_desc == "Manual":
                        await context.bot.send_message(s.active_chat_id, f"❌ {t}: {reason}")
                    continue
                
                # Check New Signal
                valid_prev, _, _ = analyze_trade(df, -2)
                is_new = not valid_prev
                
                if s.new_signals_only and not is_new:
                    if mode_desc == "Manual":
                         await context.bot.send_message(s.active_chat_id, f"⚠️ {t}: Active trade, not new")
                    continue

                # Duplicate Check for Auto Scan
                today_str = datetime.now().strftime("%Y-%m-%d")
                sig_id = f"{t}_{today_str}"
                if target == "ALL" and sig_id in s.sent_signals:
                    continue # Already sent today

                # Filters
                if d['RR'] < s.min_rr:
                    if mode_desc == "Manual": await context.bot.send_message(s.active_chat_id, f"❌ {t}: Low RR ({d['RR']:.2f})")
                    continue
                    
                atr_pct = (d['ATR']/d['P'])*100
                if atr_pct > s.max_atr:
                    if mode_desc == "Manual": await context.bot.send_message(s.active_chat_id, f"❌ {t}: High Vol ({atr_pct:.1f}%)")
                    continue

                # Position Size
                risk_amt = s.portfolio_size * (s.risk_per_trade / 100.0)
                risk_share = d['P'] - d['SL']
                if risk_share <= 0: continue
                
                shares = int(risk_amt / risk_share)
                max_shares = int(s.portfolio_size / d['P'])
                shares = min(shares, max_shares)
                
                if shares < 1:
                    if mode_desc == "Manual": await context.bot.send_message(s.active_chat_id, f"❌ {t}: Insufficient funds")
                    continue

                # SEND SIGNAL
                card_html = format_luxury_card(t, d, shares)
                await context.bot.send_message(s.active_chat_id, card_html, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                
                if target == "ALL":
                    s.sent_signals.add(sig_id)
                found_count += 1
                
            except Exception as e:
                print(f"Error scanning {t}: {e}")
                continue

        # Finish
        s.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_text = "✅ Scan Complete." if not s.stop_requested else "⏹ Scan Stopped."
        await context.bot.send_message(s.active_chat_id, f"{final_text} Found: {found_count}")
        
    except Exception as e:
        print(f"Global scan error: {e}")
    finally:
        s.is_scanning = False

# --- AUTO SCAN JOB ---
async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    # Check Market Hours (US Eastern)
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Simple check: Mon-Fri, 9:30 - 16:00
    if now.weekday() >= 5: return # Weekend
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
    if not (market_open <= now.time() <= market_close):
        BOT_STATE.next_auto_scan = f"Market Closed (Now: {now.strftime('%H:%M')} ET)"
        return

    # Check if manual scan is running
    if BOT_STATE.is_scanning:
        BOT_STATE.next_auto_scan = "Skipped (Manual Scan Active)"
        return

    # Start Auto Scan
    # We need a chat_id to send results to. Default to Admin.
    BOT_STATE.active_chat_id = ADMIN_ID 
    BOT_STATE.is_scanning = True
    BOT_STATE.stop_requested = False
    
    # Notify start
    try:
        msg = await context.bot.send_message(chat_id=ADMIN_ID, text="🤖 Starting Hourly Auto-Scan...")
        BOT_STATE.active_scan_msg_id = msg.message_id
        await perform_scan(context, "ALL")
    except Exception as e:
        print(f"Auto scan failed: {e}")

    # Calculate next run (approx)
    BOT_STATE.next_auto_scan = (now + pd.Timedelta(hours=1)).strftime("%H:%M ET")


# ==========================================
# 5. BOT RUNNER (THREADING)
# ==========================================
async def run_bot_async():
    if not TOKEN:
        print("NO TOKEN")
        return

    application = ApplicationBuilder().token(TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    
    # Job Queue
    job_queue = application.job_queue
    # Run every hour (3600 seconds)
    job_queue.run_repeating(auto_scan_job, interval=3600, first=10) 

    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    # Keep alive
    while True:
        await asyncio.sleep(1000)

def start_bot_thread():
    try:
        asyncio.run(run_bot_async())
    except Exception as e:
        print(f"Bot Loop Error: {e}")

# Use caching to run thread only once
@st.cache_resource
def launch_bot_background():
    if not TOKEN: return None
    t = threading.Thread(target=start_bot_thread, daemon=True)
    t.start()
    return t

# ==========================================
# 6. STREAMLIT SERVER DASHBOARD
# ==========================================
launch_bot_background()

st.title("🎛️ Vova Screener Server")

if not TOKEN or not ADMIN_ID:
    st.error("MISSING SECRETS: Please set TG_TOKEN and ADMIN_ID in .streamlit/secrets.toml")
    st.stop()

# Auto-refresh page every 10 seconds to show updated status
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=10000, key="dataframerefresh")

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)

status_emoji = "🟢" if BOT_STATE.is_scanning else "💤"
status_text = "SCANNING" if BOT_STATE.is_scanning else "IDLE"

col1.metric("Bot Status", f"{status_emoji} {status_text}")
col2.metric("Approved Users", len(BOT_STATE.users))
col3.metric("Last Scan", BOT_STATE.last_scan_time.split(' ')[-1] if BOT_STATE.last_scan_time != "Never" else "Never")
col4.metric("Next Auto-Scan", str(BOT_STATE.next_auto_scan))

st.markdown("---")
st.subheader("⚙️ Current Live Parameters (Bot)")

p_col1, p_col2 = st.columns(2)
with p_col1:
    st.text_input("Portfolio Size ($)", value=BOT_STATE.portfolio_size, disabled=True)
    st.text_input("Risk %", value=BOT_STATE.risk_per_trade, disabled=True)
    st.text_input("Min RR", value=BOT_STATE.min_rr, disabled=True)

with p_col2:
    st.text_input("SMA Period", value=BOT_STATE.sma_period, disabled=True)
    st.text_input("Timeframe", value=BOT_STATE.timeframe, disabled=True)
    st.text_input("New Only", value=str(BOT_STATE.new_signals_only), disabled=True)

st.info("NOTE: Parameters are currently adjustable via the Telegram Menu by the Admin. This dashboard is read-only for monitoring.")

# Debug/Manual Control (Optional)
with st.expander("Admin Operations"):
    if st.button("Reset Sent Signals Cache"):
        BOT_STATE.sent_signals.clear()
        st.success("Cache cleared.")
