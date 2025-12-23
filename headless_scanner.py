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
        else:
            c1.warning("No Auth URL")
            
        bs = "🟢 Active" if (STATE.bot_thread and STATE.bot_thread.is_alive()) else "🔴 Stopped"
        c2.metric("Bot Service", bs)

        st.subheader("🕒 Scan Status")
        ct1, ct2 = st.columns(2)
        if STATE.last_scan:
            ny = pytz.timezone('US/Eastern')
            ls = STATE.last_scan if STATE.last_scan.tzinfo else ny.localize(STATE.last_scan)
            ls_ny = ls.astimezone(ny)
            ct1.metric("Last Auto Scan (NY)", ls_ny.strftime("%H:%M:%S"))
            
            nxt = ls_ny + timedelta(hours=1)
            rem = nxt - datetime.now(ny)
            if rem.total_seconds() > 0:
                m, s = divmod(int(rem.total_seconds()), 60)
                ct2.metric("Next Scan In", f"{m}m {s}s")
            else: ct2.metric("Next Scan", "Running...")
        else:
            ct1.metric("Last Auto Scan", "N/A")
            ct2.metric("Next Scan", "Awaiting")

        st.subheader("📜 System Logs")
        with st.container(height=300):
            for l in reversed(STATE.logs[-20:]): st.text(l)
except: pass

# ==========================================
# 4. INDICATORS (EXACT WEB SCREENER LOGIC)
# ==========================================
def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ema(series, length): return series.ewm(span=length, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

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
    plus_dm, minus_dm = pd.Series(plus_dm, index=df.index), pd.Series(minus_dm, index=df.index)
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=alpha, adjust=False).mean(), plus_di, minus_di

def calc_atr(df, length):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()

# ==========================================
# 5. STRATEGY (100% EXACT WEB SCREENER LOGIC)
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        # --- Indicator Logic ---
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
        trend_st_list, seq_st_list, crit_list, peak_list, struct_list = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n
        seq_state, critical_level, seq_high, seq_low = 0, np.nan, h_arr[0], l_arr[0]
        last_confirmed_peak, last_confirmed_trough, last_peak_was_hh, last_trough_was_hl = np.nan, np.nan, False, False

        for i in range(1, n):
            c, h, l = c_arr[i], h_arr[i], l_arr[i]
            prev_st = seq_state
            is_break = False
            if prev_st == 1 and not np.isnan(critical_level): is_break = c < critical_level
            elif prev_st == -1 and not np.isnan(critical_level): is_break = c > critical_level
            
            if is_break:
                if prev_st == 1:
                    is_hh = (True if np.isnan(last_confirmed_peak) else (seq_high > last_confirmed_peak))
                    last_peak_was_hh, last_confirmed_peak = is_hh, seq_high
                    seq_state, seq_high, seq_low, critical_level = -1, h, l, h
                else:
                    is_hl = (True if np.isnan(last_confirmed_trough) else (seq_low > last_confirmed_trough))
                    last_trough_was_hl, last_confirmed_trough = is_hl, seq_low
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

            # Super Trend
            strong = (adx_v[i] > settings['adx_thresh'])
            rising = (ema_f[i] > ema_f[i-1]) and (ema_s[i] > ema_s[i-1])
            elder_bull = both_rising = rising and (h_vals[i] > h_vals[i-1])
            falling = (ema_f[i] < ema_f[i-1]) and (ema_s[i] < ema_s[i-1])
            elder_bear = falling and (h_vals[i] < h_vals[i-1])
            
            curr_t = 0
            if strong and (pdi_v[i] > mdi_v[i]) and elder_bull and (efi_vals[i] > 0): curr_t = 1
            elif strong and (mdi_v[i] > pdi_v[i]) and elder_bear and (efi_vals[i] < 0): curr_t = -1
            
            trend_st_list[i], seq_st_list[i], crit_list[i], peak_list[i], struct_list[i] = curr_t, seq_state, critical_level, last_confirmed_peak, (last_peak_was_hh and last_trough_was_hl)

        def check(idx):
            if idx < 0: return False, 0.0, np.nan, np.nan
            p, sma = c_arr[idx], df['SMA_Major'].iloc[idx]
            valid = (seq_st_list[idx] == 1) and ((p > sma) if not np.isnan(sma) else False) and (trend_st_list[idx] != -1) and struct_list[idx]
            rr, cr, pk = 0.0, crit_list[idx], peak_list[idx]
            if valid and not np.isnan(pk) and not np.isnan(cr):
                risk, reward = p - cr, pk - p
                if risk > 0 and reward > 0: rr = reward / risk
                else: valid = False
            return valid, rr, cr, pk

        v_tod, rr_t, sl_t, tp_t = check(n-1)
        v_yest, _, _, _ = check(n-2)
        
        if not v_tod or rr_t < settings['min_rr']: return None
        
        curr_c = c_arr[-1]
        curr_atr = atr_s.iloc[-1]
        atr_pct = (curr_atr / curr_c) * 100
        if atr_pct > settings['max_atr_pct']: return None
        
        # Candidate Metadata
        t_obj = yf.Ticker(ticker)
        pe = t_obj.info.get('trailingPE', 'N/A')
        if pe != 'N/A': pe = f"{pe:.2f}"
        exch = t_obj.fast_info.get('exchange', '')

        # Position Size
        risk_sh = curr_c - sl_t
        shares = 0
        if risk_sh > 0:
            risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
            shares = int(risk_amt / risk_sh)
            shares = min(shares, int(settings['portfolio_size'] / curr_c))
            if shares < 1: shares = 1

        return {
            "Ticker": ticker, "Price": curr_c, "RR": rr_t, "SL": sl_t, "TP": tp_t,
            "ATR_SL": curr_c - curr_atr, "Shares": shares, "ATR_Pct": atr_pct, 
            "Is_New": (v_tod and not v_yest), "PE": pe, "Exch": exch
        }
    except: return None

# ==========================================
# 6. BOT LOGIC & HANDLERS
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

async def help_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = (
        "ℹ️ **Справка по Vova Screener Bot**\n\n"
        "Стратегия основана на сломе структуры (Break of Structure) и трендовых фильтрах.\n\n"
        "🛠 **Описание параметров:**\n"
        "• **Portfolio**: Размер вашего капитала. Используется для расчета лота.\n"
        "• **Risk %**: Риск на одну сделку (процент потери капитала при стопе).\n"
        "• **RR**: Мин. соотношение Риск/Прибыль. Бот отсеивает сделки с низким профитом.\n"
        "• **Max ATR %**: Лимит волатильности (исключение чрезмерно резких акций).\n\n"
        "🔄 **Авто-скан**: Проверка S&P 500 каждый час (9:30-16:00 ET). Только новые сигналы."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    msg_func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Настройки Конфигурации:**\n\n"
        f"💰 **Portfolio**: ${s['portfolio_size']:,}\n"
        f"⚠️ **Risk Per Trade**: {s['risk_per_trade_pct']}%\n"
        f"📊 **Minimum RR**: {s['min_rr']}\n"
        f"📈 **Max ATR %**: {s['max_atr_pct']}%\n"
        f"🔍 **Market Mode**: {s['scan_mode']}\n"
        f"👀 **Filter**: {'🔥 Only New' if s['show_new_only'] else '✅ Show All'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port"),
         InlineKeyboardButton(f"ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Market: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥' if s['show_new_only'] else '✅'} 🔄", callback_data="ch_filt")],
        [InlineKeyboardButton("ℹ️ HELP / INFO", callback_data="show_help")]
    ]
    await msg_func(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def btn_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    uid = q.from_user.id
    s = get_settings(uid)
    d = q.data
    
    if d == "stop":
        STATE.abort_scan_users.add(uid)
        STATE.user_states.pop(uid, None)
        await q.message.reply_text("🛑 Остановка процесса...")
        return

    if d == "show_help": await help_h(u, c); return
    if d.startswith("ask_"):
        STATE.user_states[uid] = d.split("_")[1].upper()
        await q.message.reply_text(f"Введите новое значение для **{STATE.user_states[uid]}**:")
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
            await u.message.reply_text("Ввод отменен.", reply_markup=get_main_kb(uid))
        else:
            try:
                val = float(txt.replace(',', '.').replace('%', '').replace('$', '').strip())
                s = get_settings(uid)
                if st_code == "RISK": s['risk_per_trade_pct'] = val
                elif st_code == "RR": s['min_rr'] = val
                elif st_code == "PORT": s['portfolio_size'] = int(val)
                elif st_code == "ATR": s['max_atr_pct'] = val
                del STATE.user_states[uid]
                await u.message.reply_text(f"✅ Обновлено: {val}")
                await settings_menu(u, c)
                return
            except:
                await u.message.reply_text("❌ Ошибка ввода. Нужно число.")
                return

    if txt == "🚀 Запустить Скан":
        STATE.abort_scan_users.discard(uid)
        await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь": await help_h(u, c)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Авто-скан: {'✅ ВКЛ' if s['auto_scan'] else '❌ ВЫКЛ'}", reply_markup=get_main_kb(uid))
    elif txt == "/start": await u.message.reply_text("Vova Screener Bot Ready", reply_markup=get_main_kb(uid))

# --- SCAN ENGINE (FIXED STOP & 100% LOGIC) ---
async def run_scan(context, uid, s, manual=False, is_auto=False):
    if is_auto:
        last = STATE.sent_signals_cache.get("last_auto_scan_ts")
        if last and (datetime.now() - last).total_seconds() < 1800: return
        STATE.sent_signals_cache["last_auto_scan_ts"] = datetime.now()

    ticks = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(ticks)
    filt_txt = "🔥 Новые" if (s['show_new_only'] or is_auto) else "✅ Все"
    tit = "🔄 Авто" if is_auto else "🚀 Скан"
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="stop")]])
    status_msg = await context.bot.send_message(chat_id=uid, text=f"{tit}: {filt_txt}\nЗапуск...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    
    for i in range(total):
        # МГНОВЕННАЯ ПРОВЕРКА СТОПА ПЕРЕД КАЖДЫМ РАСЧЕТОМ
        if uid in STATE.abort_scan_users:
            await status_msg.edit_text(f"🛑 Прервано пользователем ({i}/{total}).")
            STATE.abort_scan_users.discard(uid)
            return
            
        t = ticks[i]
        res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
        
        if res:
            show = False
            if is_auto:
                if res['Is_New']: show = True
            else:
                show = res['Is_New'] if s['show_new_only'] else True
            
            if is_auto and show:
                if res['Ticker'] in STATE.sent_signals_cache["tickers"]: show = False
                else: STATE.sent_signals_cache["tickers"].add(res['Ticker'])

            if show:
                found += 1
                await send_sig(context, uid, res)
        
        if i % 5 == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            filled = int(10 * pct / 100)
            bar = "█"*filled + "░"*(10-filled)
            try: await status_msg.edit_text(f"{tit}: {filt_txt}\n{pct}% [{bar}] {i+1}/{total}\nНайдено: {found}", reply_markup=pkb)
            except: pass

    try: 
        await status_msg.edit_text(f"✅ {tit} завершен!\nНайдено сигналов: {found}", reply_markup=None)
        if manual: await context.bot.send_message(chat_id=uid, text="Готово! Возвращаюсь в меню.", reply_markup=get_main_kb(uid))
    except: pass

async def send_sig(ctx, uid, r):
    ticker = r['Ticker']
    tv_t = ticker.replace('-', '.')
    prefix = ""
    exch = r.get('Exch', '')
    if exch in ['NMS', 'NGM', 'NCM', 'PNK']: prefix = "NASDAQ:"
    elif exch in ['NYQ', 'NYE']: prefix = "NYSE:"
    elif exch == 'ASE': prefix = "AMEX:"
    
    full_tv = f"{prefix}{tv_t}"
    link = f"https://www.tradingview.com/chart/?symbol={full_tv}"
    ic = "🔥 NEW" if r['Is_New'] else "✅ ACTIVE"
    
    txt = (
        f"{ic} **[{full_tv}]({link})** | **Price**: ${r['Price']:.2f} (P/E: {r['PE']})\n"
        f"📊 **ATR**: {r['ATR_Pct']:.2f}% | **SL ATR**: ${r['ATR_SL']:.2f}\n"
        f"🎯 **RR**: {r['RR']:.2f} | 🛑 **SL**: ${r['SL']:.2f}\n"
        f"🏁 **TP**: ${r['TP']:.2f} | 📦 **Лот**: {r['Shares']} шт"
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
# 8. STARTUP
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
        await app.initialize()
        await app.start()
        bot_info = await app.bot.get_me()
        STATE.add_log(f"🟢 Connected: @{bot_info.username}")
        await app.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(1)

    def loop_in_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_bot())

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop_in_thread, args=(loop,), daemon=True)
    t.start()
    STATE.bot_thread = t
    return t

if __name__ == '__main__':
    if TG_TOKEN: start_bot_singleton()
