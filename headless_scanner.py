import logging
import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import pytz
from datetime import datetime, time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, JobQueue

# ==========================================
# КОНФИГУРАЦИЯ (ТВОИ ДАННЫЕ)
# ==========================================
TG_TOKEN = "8407386703:AAFtzeEQlc0H2Ev_cccHJGE3mTdxA2c_JkA"
ADMIN_ID = 1335722880  # Int format
GITHUB_USERS_URL = "https://raw.githubusercontent.com/vladimirsk1234/vova-screener/refs/heads/main/users.txt"

# Настройка логгирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. LOGIC & MATH (EXACT COPY FROM SCREENER)
# ==========================================

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

# --- VOVA STRATEGY LOGIC ---
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
# 3. STATE & USER MANAGEMENT
# ==========================================

# Значения по умолчанию
DEFAULT_PARAMS = {
    'portfolio': 10000,
    'min_rr': 1.25,
    'risk_pct': 0.2,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True
}

# Хранилище настроек пользователей: { chat_id: {params} }
user_configs = {}
# Хранилище отправленных сигналов для анти-спама: { date_str: {ticker1, ticker2} }
sent_signals = {}

def get_user_params(chat_id):
    if chat_id not in user_configs:
        user_configs[chat_id] = DEFAULT_PARAMS.copy()
    return user_configs[chat_id]

async def check_auth(username, user_id):
    if str(user_id) == str(ADMIN_ID): return True
    try:
        resp = requests.get(GITHUB_USERS_URL)
        if resp.status_code == 200:
            allowed_users = [u.strip() for u in resp.text.splitlines() if u.strip()]
            return username in allowed_users
    except:
        return False
    return False

# ==========================================
# 4. BOT HELPERS & FORMATTING
# ==========================================

def format_card(ticker, d, shares, is_new, pe_val):
    """
    Создает COOL LOOKING LUXURY CARD в HTML.
    """
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    ticker_html = f"<a href='{tv_link}'><b>{ticker}</b></a>"
    new_badge = "🆕" if is_new else ""
    
    val_pos = shares * d['P']
    profit = (d['TP'] - d['P']) * shares
    loss = (d['P'] - d['SL']) * shares
    atr_pct = (d['ATR'] / d['P']) * 100
    pe_str = f"(PE: {pe_val:.0f})" if pe_val else ""

    # Compact HTML Layout
    msg = (
        f"💎 {ticker_html} {new_badge}\n"
        f"💵 <b>${d['P']:.2f}</b> {pe_str}\n"
        f"📉 Stop: <b>{d['SL']:.2f}</b> | 🎯 Target: <b>{d['TP']:.2f}</b>\n"
        f"⚖️ R:R: <b>{d['RR']:.2f}</b> | 💼 Pos: <b>{shares}</b> (${val_pos:.0f})\n"
        f"📊 ATR: ${d['ATR']:.2f} ({atr_pct:.1f}%) | 🧱 Crit: {d['Crit']:.2f}"
    )
    return msg

def is_market_open():
    """Проверяет, открыт ли рынок США (9:30 - 16:00 ET, Пн-Пт)."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # 0 = Monday, 4 = Friday
    if now.weekday() > 4: return False
    
    market_start = time(9, 30)
    market_end = time(16, 0)
    return market_start <= now.time() <= market_end

# ==========================================
# 5. CORE SCANNING FUNCTION
# ==========================================
async def perform_scan(context: ContextTypes.DEFAULT_TYPE, chat_id: int, custom_tickers=None, auto_mode=False):
    params = get_user_params(chat_id)
    
    # Snapshot params at start of scan
    p_src = "Custom" if custom_tickers else "All S&P 500"
    p_sma = params['sma']
    p_tf = params['tf']
    p_rr = params['min_rr']
    p_risk = params['risk_pct']
    p_matr = params['max_atr']
    p_port = params['portfolio']
    p_new = params['new_only']

    # CONSTANTS
    EMA_F=20; EMA_S=40; ADX_L=14; ADX_T=20; ATR_L=14

    if custom_tickers:
        tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    else:
        tickers = get_sp500_tickers()

    if not tickers:
        await context.bot.send_message(chat_id, "⚠️ Нет тикеров для сканирования.")
        return

    # Info message only for manual scan
    if not auto_mode:
        await context.bot.send_message(chat_id, f"🚀 <b>Start Scanning</b> ({len(tickers)} tickers)\n⚙️ TF: {p_tf}, SMA: {p_sma}", parse_mode=ParseMode.HTML)

    found_count = 0
    today_str = datetime.now().strftime('%Y-%m-%d')
    if today_str not in sent_signals: sent_signals[today_str] = set()

    for t in tickers:
        # Check if user stopped scan (only relevant for long loops, tricky in async, simpler to let it run per ticker)
        # For auto_mode, we enforce duplication check
        if auto_mode:
            unique_id = f"{chat_id}_{t}"
            if unique_id in sent_signals[today_str]:
                continue

        try:
            inter = "1d" if p_tf == "Daily" else "1wk"
            fetch_period = "2y" if p_tf == "Daily" else "5y"
            
            # Using threads for blocking IO
            df = await asyncio.to_thread(yf.download, t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            if len(df) < p_sma + 5: continue

            # Logic
            df = run_vova_logic(df, p_sma, EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # Analyze Current
            valid, d, reason = analyze_trade(df, -1)
            
            # Reasons to reject if specific single ticker request? 
            if custom_tickers and not valid:
                 await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: {reason}", parse_mode=ParseMode.HTML)
                 continue

            if not valid: continue

            # Filters
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            # If Auto or specific setting, skip old signals
            if p_new and not is_new: continue

            if d['RR'] < p_rr: 
                if custom_tickers: await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: Low RR ({d['RR']:.2f})", parse_mode=ParseMode.HTML)
                continue
            
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > p_matr: 
                if custom_tickers: await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: High Vol ({atr_pct:.1f}%)", parse_mode=ParseMode.HTML)
                continue

            # Position Sizing
            risk_amt = p_port * (p_risk / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            
            shares = int(risk_amt / risk_share)
            max_shares_portfolio = int(p_port / d['P'])
            shares = min(shares, max_shares_portfolio)
            
            if shares < 1: 
                if custom_tickers: await context.bot.send_message(chat_id, f"❌ <b>{t}</b>: Low Funds", parse_mode=ParseMode.HTML)
                continue

            # Success! Fetch PE and Send
            pe = await asyncio.to_thread(get_financial_info, t)
            
            card_html = format_card(t, d, shares, is_new, pe)
            
            # Send immediately
            await context.bot.send_message(chat_id, card_html, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            
            found_count += 1
            
            # Mark as sent for today
            if auto_mode:
                sent_signals[today_str].add(f"{chat_id}_{t}")

        except Exception as e:
            logger.error(f"Scan error {t}: {e}")
            continue

    if not auto_mode and not custom_tickers:
        await context.bot.send_message(chat_id, f"🏁 <b>Scan Finished.</b> Found: {found_count}", parse_mode=ParseMode.HTML)


# ==========================================
# 6. TELEGRAM HANDLERS
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not await check_auth(user.username, user.id):
        await update.message.reply_text("⛔ Доступ запрещен.")
        return
    
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я Screener Vova Bot. Я использую твою стратегию на 100%.\n\n"
        "<b>Команды:</b>\n"
        "/scan - Запустить ручной скан (S&P 500)\n"
        "/scan AAPL,TSLA - Проверить конкретные тикеры\n"
        "/auto - Вкл/Выкл автоматический скан (каждый час)\n"
        "/settings - Показать настройки\n"
        "/set_risk 0.5 - Изменить риск на сделку\n"
        "/set_rr 1.5 - Изменить мин. RR\n"
        "/set_port 20000 - Изменить размер портфеля\n"
        "/set_tf Daily - (Daily/Weekly)\n"
        "/set_sma 150 - (100/150/200)",
        parse_mode=ParseMode.HTML
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update.effective_user.username, update.effective_user.id): return
    
    args = context.args
    custom_tickers = "".join(args) if args else None
    
    # Async scan trigger
    asyncio.create_task(perform_scan(context, update.effective_chat.id, custom_tickers, auto_mode=False))

# --- SETTINGS HANDLERS ---
async def settings_view(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = get_user_params(update.effective_chat.id)
    msg = (
        "<b>⚙️ CURRENT SETTINGS:</b>\n"
        f"💰 Portfolio: ${p['portfolio']}\n"
        f"⚖️ Min RR: {p['min_rr']}\n"
        f"⚠️ Risk: {p['risk_pct']}%\n"
        f"📊 Max ATR: {p['max_atr']}%\n"
        f"📈 SMA: {p['sma']}\n"
        f"📅 TF: {p['tf']}\n"
        f"🆕 New Only: {p['new_only']}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def set_param(update: Update, context: ContextTypes.DEFAULT_TYPE, key, type_conv):
    if not context.args:
        await update.message.reply_text(f"❌ Укажите значение. Пример: /set_{key} value")
        return
    try:
        val = type_conv(context.args[0])
        get_user_params(update.effective_chat.id)[key] = val
        await update.message.reply_text(f"✅ {key} set to {val}")
    except:
        await update.message.reply_text("❌ Ошибка формата.")

async def cmd_set_port(u, c): await set_param(u, c, 'portfolio', int)
async def cmd_set_rr(u, c): await set_param(u, c, 'min_rr', float)
async def cmd_set_risk(u, c): await set_param(u, c, 'risk_pct', float)
async def cmd_set_sma(u, c): await set_param(u, c, 'sma', int)

async def cmd_set_tf(u, c):
    val = c.args[0].capitalize()
    if val in ['Daily', 'Weekly']:
        get_user_params(u.effective_chat.id)['tf'] = val
        await u.message.reply_text(f"✅ Timeframe set to {val}")
    else:
        await u.message.reply_text("❌ Use Daily or Weekly")

# --- AUTO SCAN JOBS ---
async def hourly_scan_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    
    # 1. Check US Market Open
    if not is_market_open():
        # Можно раскомментировать для дебага, но лучше молчать, чтобы не спамить
        # await context.bot.send_message(chat_id, "💤 Рынок закрыт. Сплю.") 
        return

    # 2. Perform Scan
    await perform_scan(context, chat_id, custom_tickers=None, auto_mode=True)

async def toggle_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job_name = f"auto_scan_{chat_id}"
    
    current_jobs = context.job_queue.get_jobs_by_name(job_name)
    
    if current_jobs:
        for job in current_jobs: job.schedule_removal()
        await update.message.reply_text("⏹ <b>Auto Scan STOPPED.</b>", parse_mode=ParseMode.HTML)
    else:
        # Check if manual scan was done at least once? (User requirement)
        # We assume if they know the command, they are ready. 
        # Schedule every hour (3600 seconds)
        context.job_queue.run_repeating(hourly_scan_job, interval=3600, first=10, chat_id=chat_id, name=job_name)
        await update.message.reply_text("▶️ <b>Auto Scan STARTED.</b>\nChecking every hour during US Market Open.", parse_mode=ParseMode.HTML)

# ==========================================
# 7. MAIN RUN
# ==========================================

def main():
    application = Application.builder().token(TG_TOKEN).build()
    
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("auto", toggle_auto))
    application.add_handler(CommandHandler("settings", settings_view))
    
    # Setters
    application.add_handler(CommandHandler("set_port", cmd_set_port))
    application.add_handler(CommandHandler("set_rr", cmd_set_rr))
    application.add_handler(CommandHandler("set_risk", cmd_set_risk))
    application.add_handler(CommandHandler("set_sma", cmd_set_sma))
    application.add_handler(CommandHandler("set_tf", cmd_set_tf))
    
    # Start
    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None, close_loop=False)

if __name__ == "__main__":
    main()

