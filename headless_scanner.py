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

# Применяем патч для asyncio, чтобы Streamlit и Telegram Bot работали вместе
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
                    c1.metric("✅ Пользователи", f"{len(ul)}")
                else: c1.error(f"GH Error: {r.status_code}")
            except: c1.error("Net Error")
        else:
            c1.warning("No GH URL")
            
        bs = "🟢 Активен" if (STATE.bot_thread and STATE.bot_thread.is_alive()) else "🔴 Остановлен"
        c2.metric("Сервис Бота", bs)

        st.subheader("🕒 Статус Сканирования")
        ct1, ct2 = st.columns(2)
        if STATE.last_scan:
            ny = pytz.timezone('US/Eastern')
            ls = STATE.last_scan if STATE.last_scan.tzinfo else ny.localize(STATE.last_scan)
            ls_ny = ls.astimezone(ny)
            ct1.metric("Последний Авто-Скан (NY)", ls_ny.strftime("%H:%M:%S"))
            
            nxt = ls_ny + timedelta(hours=1)
            now_ny = datetime.now(ny)
            rem = nxt - now_ny
            if rem.total_seconds() > 0:
                m, s = divmod(int(rem.total_seconds()), 60)
                ct2.metric("След. проверка через", f"{m}м {s}с")
            else:
                ct2.metric("След. проверка через", "Запуск...")
        else:
            ct1.metric("Последний запуск", "Нет данных")
            ct2.metric("След. проверка", "Ожидание...")

        st.subheader("📜 Логи Системы")
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

def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ema(series, length): return series.ewm(span=length, adjust=False).mean()
def calc_atr(df, length):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_adx(df, length):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm, minus_dm = pd.Series(plus_dm, index=df.index), pd.Series(minus_dm, index=df.index)
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    mdi = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    dx = 100 * ((pdi - mdi).abs() / (pdi + mdi))
    return dx.ewm(alpha=alpha, adjust=False).mean(), pdi, mdi

# ==========================================
# 5. STRATEGY (100% EXACT LOGIC)
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
        trend_st_list, seq_st_list, crit_list, peak_list, struct_list = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n
        seq_st, crit, s_h, s_l = 0, np.nan, h_arr[0], l_arr[0]
        l_pk, l_tr, l_hh, l_hl = np.nan, np.nan, False, False

        for i in range(1, n):
            c, h, l = c_arr[i], h_arr[i], l_arr[i]
            prev_st, is_brk = seq_st, False
            if prev_st == 1 and not np.isnan(crit): is_brk = c < crit
            elif prev_st == -1 and not np.isnan(crit): is_brk = c > crit
            
            if is_brk:
                if prev_st == 1:
                    l_hh, l_pk = (True if np.isnan(l_pk) else (s_h > l_pk)), s_h
                    seq_st, s_h, s_l, crit = -1, h, l, h
                else:
                    l_hl, l_tr = (True if np.isnan(l_tr) else (s_l > l_tr)), s_l
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
            
            curr_t = 0
            if strong and (pdi_v[i] > mdi_v[i]) and rising and (h_vals[i] > h_vals[i-1]) and (efi_vals[i] > 0): curr_t = 1
            elif strong and (mdi_v[i] > pdi_v[i]) and falling and (h_vals[i] < h_vals[i-1]) and (efi_vals[i] < 0): curr_t = -1
            
            trend_st_list[i], seq_st_list[i], crit_list[i], peak_list[i], struct_list[i] = curr_t, seq_st, crit, l_pk, (l_hh and l_hl)

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
        
        # Медленные данные только для кандидатов
        t_obj = yf.Ticker(ticker)
        pe = t_obj.info.get('trailingPE', 'N/A')
        if pe != 'N/A': pe = f"{pe:.2f}"
        exch = t_obj.fast_info.get('exchange', '')

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
        await u.message.reply_text(f"⛔ Доступ запрещен. ID: `{uid}`")
        return
    await u.message.reply_text("👋 **Vova Screener Bot**", reply_markup=get_main_kb(uid))

async def help_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = (
        "ℹ️ **Справка по боту**\n\n"
        "**Стратегия:** Break of Structure + SuperTrend (фильтр SMA 200).\n\n"
        "• **Portfolio**: Весь ваш капитал ($).\n"
        "• **Risk %**: Риск на одну сделку.\n"
        "• **RR**: Мин. соотношение Прибыль/Риск.\n"
        "• **Max ATR %**: Лимит волатильности.\n\n"
        "🔄 **Авто-скан**: Каждый час при открытом рынке США."
    )
    await u.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Настройки:**\n\n"
        f"💰 **Portfolio**: ${s['portfolio_size']:,}\n"
        f"⚠️ **Risk Per Trade**: {s['risk_per_trade_pct']}%\n"
        f"📊 **Minimum RR**: {s['min_rr']}\n"
        f"📈 **Max ATR Limit**: {s['max_atr_pct']}%\n"
        f"🔍 **Market Mode**: {s['scan_mode']}\n"
        f"👀 **Фильтр**: {'🔥 Только новые' if s['show_new_only'] else '✅ Все активные'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port"),
         InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Market: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥 Новые' if s['show_new_only'] else '✅ Все'} 🔄", callback_data="ch_filt")],
        [InlineKeyboardButton("ℹ️ HELP / INFO", callback_data="show_help")]
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
        STATE.user_states.pop(uid, None) # Сброс состояния ввода
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
            await u.message.reply_text("Отменено.", reply_markup=get_main_kb(uid))
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
                await u.message.reply_text("❌ Ошибка ввода числа.")
                return

    if txt == "🚀 Запустить Скан": await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь": await help_h(u, c)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Авто-скан: {'✅ ВКЛ' if s['auto_scan'] else '❌ ВЫКЛ'}", reply_markup=get_main_kb(uid))
    elif txt == "/start": await start_h(u, c)

# --- SCAN ENGINE ---
async def run_scan(context, uid, s, manual=False, is_auto=False):
    if is_auto:
        last = STATE.sent_signals_cache.get("last_auto_scan_ts")
        if last and (datetime.now() - last).total_seconds() < 1800: return
        STATE.sent_signals_cache["last_auto_scan_ts"] = datetime.now()

    if uid in STATE.abort_scan_users: STATE.abort_scan_users.remove(uid)
    
    ticks = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(ticks)
    filt_txt = "🔥 Новые" if (s['show_new_only'] or is_auto) else "✅ Все"
    tit = "🔄 Авто" if is_auto else "🚀 Скан"
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="stop")]])
    status_msg = await context.bot.send_message(chat_id=uid, text=f"{tit}: {filt_txt}\nЗапуск...", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    
    for i in range(total):
        if uid in STATE.abort_scan_users:
            await status_msg.edit_text(f"🛑 Прервано пользователем ({i}/{total}).")
            STATE.abort_scan_users.remove(uid)
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
        if manual: await context.bot.send_message(chat_id=uid, text="Готово.", reply_markup=get_main_kb(uid))
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
        f"{ic} **[{full_tv}]({link})** | **Price**: ${r['Price']:.2f} | **P/E**: {r['PE']}\n"
        f"📊 **ATR**: {r['ATR_Pct']:.2f}% | **SL ATR**: ${r['ATR_SL']:.2f}\n"
        f"🎯 **RR**: {r['RR']:.2f} | 🛑 **SL**: ${r['SL']:.2f}\n"
        f"🏁 **TP**: ${r['TP']:.2f} | 📦 **Size**: {r['Shares']} шт"
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
        STATE.add_log(f"💤 Рынок закрыт")

# ==========================================
# 8. STARTUP (STABLE SINGLETON)
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
        
        # Инициализация и запуск приложения
        await app.initialize()
        await app.start()
        
        # Проверка соединения
        bot_info = await app.bot.get_me()
        STATE.add_log(f"🟢 Бот подключен: @{bot_info.username}")
        
        await app.updater.start_polling(drop_pending_updates=True)
        
        # Бесконечный цикл для поддержания работы потока
        while True:
            await asyncio.sleep(1)

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
