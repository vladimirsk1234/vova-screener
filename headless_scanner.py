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
from streamlit_autorefresh import st_autorefresh
import nest_asyncio

# Применяем патч для асинхронности в среде Streamlit
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
        self.active_scans = set()
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
# 3. МОНИТОРИНГ (STREAMLIT)
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
                r = requests.get(GITHUB_USERS_URL, timeout=5)
                if r.status_code == 200:
                    ul = [l for l in r.text.splitlines() if l.strip()]
                    c1.metric("✅ Юзеров одобрено", f"{len(ul)}")
                else: c1.error(f"GitHub Error: {r.status_code}")
            except: c1.error("Ошибка сети GitHub")
            
        is_bot_alive = STATE.bot_thread is not None and STATE.bot_thread.is_alive()
        c2.metric("Статус Бота", "🟢 Активен" if is_bot_alive else "🔴 Остановлен")

        st.subheader("🕒 Статус Сканера")
        ct1, ct2 = st.columns(2)
        if STATE.last_scan:
            ny = pytz.timezone('US/Eastern')
            ls = STATE.last_scan if STATE.last_scan.tzinfo else ny.localize(STATE.last_scan)
            ls_ny = ls.astimezone(ny)
            ct1.metric("Последний запуск (NY)", ls_ny.strftime("%H:%M:%S"))
            
            nxt = ls_ny + timedelta(hours=1)
            rem = nxt - datetime.now(ny)
            if rem.total_seconds() > 0:
                m, s = divmod(int(rem.total_seconds()), 60)
                ct2.metric("До след. запуска", f"{m}м {s}с")
            else: ct2.metric("До след. запуска", "Запуск...")
        else:
            ct1.metric("Последний запуск", "Ожидание")
            ct2.metric("До след. запуска", "Скоро")

        st.subheader("📜 Логи системы")
        with st.container(height=300):
            for l in reversed(STATE.logs[-20:]): st.text(l)
except: pass

# ==========================================
# 4. ИНДИКАТОРЫ (100% ИЗ ВЕБ-СКРИНЕРА)
# ==========================================
def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ema(series, length): return series.ewm(span=length, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

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
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=alpha, adjust=False).mean(), plus_di, minus_di

def calc_atr(df, length):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()

# ==========================================
# 5. СТРАТЕГИЯ (100% ИДЕНТИЧНАЯ ЛОГИКА)
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False, timeout=10)
        if df.empty or len(df) < settings['len_major']: return None

        # Расчет индикаторов точно как в веб-версии
        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx_series, plus_di, minus_di = calc_adx(df, settings['adx_len'])
        atr_series = calc_atr(df, settings['atr_len'])
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        change = df['Close'].diff()
        efi_raw = change * df['Volume']
        df['EFI'] = calc_ema(efi_raw, settings['len_fast'])

        n = len(df)
        trend_state_list, seq_state_list, critical_level_list, peak_list, struct_ok_list = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n
        seq_state, critical_level, seq_high, seq_low = 0, np.nan, df['High'].iloc[0], df['Low'].iloc[0]
        last_confirmed_peak, last_confirmed_trough, last_peak_was_hh, last_trough_was_hl = np.nan, np.nan, False, False

        close_arr, high_arr, low_arr = df['Close'].values, df['High'].values, df['Low'].values
        ema_fast_vals, ema_slow_vals, macd_hist_vals, efi_vals = df['EMA_Fast'].values, df['EMA_Slow'].values, macd_hist.values, df['EFI'].values
        adx_vals, pdi_vals, mdi_vals = adx_series.values, plus_di.values, minus_di.values

        for i in range(1, n):
            c, h, l = close_arr[i], high_arr[i], low_arr[i]
            prev_seq_state, is_break = seq_state, False
            if prev_seq_state == 1:
                if not np.isnan(critical_level): is_break = c < critical_level 
            elif prev_seq_state == -1:
                if not np.isnan(critical_level): is_break = c > critical_level 
            
            if is_break:
                if prev_seq_state == 1:
                    is_current_peak_hh = True if np.isnan(last_confirmed_peak) else (seq_high > last_confirmed_peak)
                    last_peak_was_hh, last_confirmed_peak = is_current_peak_hh, seq_high
                    seq_state, seq_high, seq_low, critical_level = -1, h, l, h
                else:
                    is_current_trough_hl = True if np.isnan(last_confirmed_trough) else (seq_low > last_confirmed_trough)
                    last_trough_was_hl, last_confirmed_trough = is_current_trough_hl, seq_low
                    seq_state, seq_high, seq_low, critical_level = 1, h, l, l
            else:
                if seq_state == 1:
                    if h >= seq_high: seq_high, critical_level = h, l
                elif seq_state == -1:
                    if l <= seq_low: seq_low, critical_level = l, h
                else:
                    if c > seq_high: seq_state, critical_level = 1, l
                    elif c < seq_low: seq_state, critical_level = -1, h
                    else: seq_high, seq_low = max(seq_high, h), min(seq_low, l)

            # Super Trend Logic
            adx_strong = (adx_vals[i] > settings['adx_thresh'])
            both_rising = (ema_fast_vals[i] > ema_fast_vals[i-1]) and (ema_slow_vals[i] > ema_slow_vals[i-1])
            elder_bull = both_rising and (macd_hist_vals[i] > macd_hist_vals[i-1])
            both_falling = (ema_fast_vals[i] < ema_fast_vals[i-1]) and (ema_slow_vals[i] < ema_slow_vals[i-1])
            elder_bear = both_falling and (macd_hist_vals[i] < macd_hist_vals[i-1])
            
            curr_trend_state = 0
            if adx_strong and (pdi_vals[i] > mdi_vals[i]) and elder_bull and (efi_vals[i] > 0):
                curr_trend_state = 1
            elif adx_strong and (mdi_vals[i] > pdi_vals[i]) and elder_bear and (efi_vals[i] < 0):
                curr_trend_state = -1
            
            trend_state_list[i], seq_state_list[i], critical_level_list[i], peak_list[i], struct_ok_list[i] = curr_trend_state, seq_state, critical_level, last_confirmed_peak, (last_peak_was_hh and last_trough_was_hl)

        def check_conditions(idx):
            if idx < 0: return False, 0.0, np.nan, np.nan
            price, sma = close_arr[idx], df['SMA_Major'].iloc[idx]
            s_state, t_state, is_struct_ok = seq_state_list[idx], trend_state_list[idx], struct_ok_list[idx]
            crit, peak = critical_level_list[idx], peak_list[idx]
            # Valid if: UP sequence + Above MA + Trend NOT bear + Structure confirmed
            valid = (s_state == 1) and (price > sma) and (t_state != -1) and is_struct_ok
            rr = 0.0
            if valid and not np.isnan(peak) and not np.isnan(crit):
                risk, reward = price - crit, peak - price
                if risk > 0 and reward > 0: rr = reward / risk
                else: valid = False
            return valid, rr, crit, peak

        v_tod, rr_t, sl_t, tp_t = check_conditions(n-1)
        v_yest, _, _, _ = check_conditions(n-2)
        
        if not v_tod or rr_t < settings['min_rr']: return None
        
        curr_c = close_arr[-1]
        atr_val = atr_series.iloc[-1]
        atr_pct = (atr_val / curr_c) * 100
        if atr_pct > settings['max_atr_pct']: return None
        
        return {
            "Ticker": ticker, "Price": curr_c, "RR": rr_t, "SL": sl_t, "TP": tp_t,
            "ATR_SL": curr_c - atr_val, "ATR_Pct": atr_pct, "Is_New": (v_tod and not v_yest)
        }
    except: return None

# ==========================================
# 6. МЕТАДАННЫЕ (P/E и Биржа)
# ==========================================
def fetch_meta(ticker):
    try:
        t = yf.Ticker(ticker)
        pe = t.info.get('trailingPE', 'N/A')
        pe = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
        ex = t.fast_info.get('exchange', '')
        return {"PE": pe, "Exch": ex}
    except: return {"PE": "N/A", "Exch": ""}

# ==========================================
# 7. ХЕНДЛЕРЫ БОТА
# ==========================================
async def check_auth_async(uid):
    if ADMIN_ID and str(uid) == str(ADMIN_ID): return True
    if not GITHUB_USERS_URL: return False
    try:
        r = await asyncio.get_running_loop().run_in_executor(None, requests.get, GITHUB_USERS_URL)
        return (str(uid) in [l.strip() for l in r.text.splitlines() if l.strip()])
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
    await u.message.reply_text("👋 **Vova Screener Bot**\nЛогика 100% как в вебе.", reply_markup=get_main_kb(uid))

async def settings_menu(u: Update, c: ContextTypes.DEFAULT_TYPE):
    uid = u.effective_user.id if not u.callback_query else u.callback_query.from_user.id
    func = u.callback_query.edit_message_text if u.callback_query else u.message.reply_text
    s = get_settings(uid)
    txt = (
        f"⚙️ **Настройки бота:**\n\n"
        f"💰 **Portfolio**: ${s['portfolio_size']:,}\n"
        f"⚠️ **Risk %**: {s['risk_per_trade_pct']}%\n"
        f"📊 **Min RR**: {s['min_rr']}\n"
        f"🔍 **Market**: {s['scan_mode']}\n"
        f"👀 **Filt**: {'🔥 Только новые' if s['show_new_only'] else '✅ Все активные'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Port: ${s['portfolio_size']} ✏️", callback_data="ask_port"),
         InlineKeyboardButton(f"ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Market: {s['scan_mode']} 🔄", callback_data="ch_mode"),
         InlineKeyboardButton(f"Filt: {'🔥' if s['show_new_only'] else '✅'} 🔄", callback_data="ch_filt")]
    ]
    await func(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def btn_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    uid = q.from_user.id
    s = get_settings(uid)
    if q.data == "stop":
        STATE.abort_scan_users.add(uid)
        await q.message.reply_text("🛑 Остановка... Процесс будет прерван немедленно.")
        return
    if q.data.startswith("ask_"):
        STATE.user_states[uid] = q.data.split("_")[1].upper()
        await q.message.reply_text(f"Введите новое значение для {STATE.user_states[uid]}:")
        return
    if q.data == "ch_mode": s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
    elif q.data == "ch_filt": s['show_new_only'] = not s['show_new_only']
    await settings_menu(u, c)

async def txt_h(u: Update, c: ContextTypes.DEFAULT_TYPE):
    txt = u.message.text
    uid = u.effective_user.id
    if not await check_auth_async(uid): return

    if uid in STATE.user_states:
        st_code = STATE.user_states.pop(uid)
        try:
            val = float(txt.replace(',', '.').replace('%', '').replace('$', '').strip())
            s = get_settings(uid)
            if st_code == "RISK": s['risk_per_trade_pct'] = val
            elif st_code == "RR": s['min_rr'] = val
            elif st_code == "PORT": s['portfolio_size'] = int(val)
            elif st_code == "ATR": s['max_atr_pct'] = val
            await u.message.reply_text(f"✅ Обновлено: {val}")
            await settings_menu(u, c)
        except: await u.message.reply_text("❌ Ошибка ввода.")
        return

    if txt == "🚀 Запустить Скан":
        if uid in STATE.active_scans:
            await u.message.reply_text("⚠️ Сканирование уже запущено!")
            return
        await run_scan(c, uid, get_settings(uid), manual=True)
    elif txt == "⚙️ Настройки": await settings_menu(u, c)
    elif txt == "ℹ️ Помощь":
        await u.message.reply_text("Бот ищет сигналы на покупку акций США по стратегии Vova (BOS + SuperTrend).", parse_mode=ParseMode.MARKDOWN)
    elif txt.startswith("🔄 Авто"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await u.message.reply_text(f"Авто-скан: {'✅ ВКЛ' if s['auto_scan'] else '❌ ВЫКЛ'}", reply_markup=get_main_kb(uid))
    elif txt == "/start": await start_h(u, c)

# --- SCAN ENGINE (FIXED STOP & 100% LOGIC) ---
async def run_scan(ctx, uid, s, manual=False, is_auto=False):
    if is_auto:
        last = STATE.sent_signals_cache.get("last_auto_scan_ts")
        if last and (datetime.now() - last).total_seconds() < 1800: return
        STATE.sent_signals_cache["last_auto_scan_ts"] = datetime.now()

    STATE.active_scans.add(uid)
    STATE.abort_scan_users.discard(uid)
    
    try:
        ticks = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
        total = len(ticks)
        filt_txt = "🔥 Новые" if (s['show_new_only'] or is_auto) else "✅ Все"
        tit = "🔄 Авто" if is_auto else "🚀 Скан"
        
        pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="stop")]])
        status = await ctx.bot.send_message(uid, f"{tit}: {filt_txt}\nЗапуск...", reply_markup=pkb)
        
        loop = asyncio.get_running_loop()
        found = 0
        for i, t in enumerate(ticks):
            # КРИТИЧЕСКАЯ ПРОВЕРКА СТОПА
            if uid in STATE.abort_scan_users:
                await status.edit_text(f"🛑 Прервано пользователем на {i}/{total}.")
                return
                
            res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
            if res:
                show = res['Is_New'] if (is_auto or s['show_new_only']) else True
                if is_auto and show and res['Ticker'] in STATE.sent_signals_cache["tickers"]: show = False
                if show:
                    if is_auto: STATE.sent_signals_cache["tickers"].add(res['Ticker'])
                    found += 1
                    meta = await loop.run_in_executor(None, fetch_meta, t)
                    res.update(meta)
                    await send_sig(ctx, uid, res)
            
            if i % 5 == 0 or i == total - 1:
                pct = int((i + 1) / total * 100)
                filled = int(10 * pct / 100)
                bar = "█"*filled + "░"*(10-filled)
                try: await status.edit_text(f"{tit}: {filt_txt}\n{pct}% [{bar}] {i+1}/{total}\nНайдено: {found}", reply_markup=pkb)
                except: pass
                
        await status.edit_text(f"✅ {tit} завершен! Найдено: {found}", reply_markup=None)
        if manual: await ctx.bot.send_message(uid, "Готово.", reply_markup=get_main_kb(uid))
    finally:
        STATE.active_scans.discard(uid)
        STATE.abort_scan_users.discard(uid)

async def send_sig(ctx, uid, r):
    ticker = r['Ticker']
    tv_t = ticker.replace('-', '.')
    pre = "NASDAQ:" if r['Exch'] in ['NMS', 'NGM', 'NCM'] else "NYSE:" if r['Exch'] in ['NYQ', 'NYE'] else ""
    full = f"{pre}{tv_t}"
    link = f"https://www.tradingview.com/chart/?symbol={full}"
    ic = "🔥 NEW" if r['Is_New'] else "✅ ACTIVE"
    
    s = get_settings(uid)
    risk_amt = s['portfolio_size'] * (s['risk_per_trade_pct'] / 100.0)
    risk_per_sh = r['Price'] - r['SL']
    shares = int(risk_amt / risk_per_sh) if risk_per_sh > 0 else 0
    shares = min(shares, int(s['portfolio_size'] / r['Price']))

    txt = (
        f"{ic} **[{full}]({link})** | **${r['Price']:.2f}** (P/E: {r['PE']})\n"
        f"📊 **ATR**: {r['ATR_Pct']:.2f}% | **SL ATR**: ${r['ATR_SL']:.2f}\n"
        f"🎯 **RR**: {r['RR']:.2f} | 🛑 **SL**: ${r['SL']:.2f}\n"
        f"🏁 **TP**: ${r['TP']:.2f} | 📦 **Лот**: {shares} шт"
    )
    await ctx.bot.send_message(uid, txt, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def auto_job(ctx: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    STATE.last_scan = now
    if STATE.sent_signals_cache["date"] != now.strftime("%Y-%m-%d"):
        STATE.sent_signals_cache.update({"date": now.strftime("%Y-%m-%d"), "tickers": set()})
    
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        STATE.add_log(f"🔄 Авто-сканирование запущено")
        for uid, s in STATE.user_settings.items():
            if s.get('auto_scan', False): await run_scan(ctx, uid, s, manual=False, is_auto=True)
    else: STATE.add_log(f"💤 Рынок закрыт")

@st.cache_resource
def start_bot_singleton():
    if not TG_TOKEN: return None
    async def run_bot():
        app = ApplicationBuilder().token(TG_TOKEN).build()
        app.add_handler(CommandHandler('start', start_h))
        app.add_handler(CallbackQueryHandler(btn_h))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), txt_h))
        app.job_queue.run_repeating(auto_job, interval=3600, first=10)
        await app.initialize(); await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(10)
    
    def thread_runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_bot())

    t = threading.Thread(target=thread_runner, daemon=True)
    t.start()
    STATE.bot_thread = t
    return True

if __name__ == '__main__':
    if TG_TOKEN: start_bot_singleton()
