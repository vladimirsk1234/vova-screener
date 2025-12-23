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
from streamlit_autorefresh import st_autorefresh # Для автообновления UI

# === FIX FOR STREAMLIT ASYNCIO CONFLICT ===
import nest_asyncio
nest_asyncio.apply()
# ==========================================

# ==========================================
# 0. ГЛОБАЛЬНОЕ СОСТОЯНИЕ (Для UI)
# ==========================================
if 'BOT_STATE' not in globals():
    globals()['BOT_STATE'] = {
        "last_scan": None,
        "logs": []
    }
BOT_STATE = globals()['BOT_STATE']

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕКРЕТЫ
# ==========================================

try:
    import streamlit as st
    try:
        if __name__ == '__main__':
            st_autorefresh(interval=10000, key="monitor_refresh")
            st.title("🤖 Vova Screener Bot Monitor")
            
            tg_token_check = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
            gh_url_check = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
            
            col_u1, col_u2 = st.columns(2)
            if gh_url_check:
                try:
                    resp = requests.get(gh_url_check)
                    if resp.status_code == 200:
                        users_list = [l for l in resp.text.splitlines() if l.strip()]
                        col_u1.metric("✅ Авторизовано", f"{len(users_list)} юзеров")
                    else:
                        col_u1.error(f"GitHub Error: {resp.status_code}")
                except: col_u1.error("Ошибка сети")
            else:
                col_u1.warning("GitHub URL не задан")
            
            col_u2.metric("Статус Бота", "🟢 Работает" if tg_token_check else "🔴 Нет токена")

            st.subheader("🕒 Статус Сканера")
            col_t1, col_t2 = st.columns(2)
            last_scan_time = BOT_STATE.get("last_scan")
            if last_scan_time:
                col_t1.metric("Последний авто-скан", last_scan_time.strftime("%H:%M:%S (NY)"))
                next_scan_time = last_scan_time + timedelta(hours=1)
                delta = next_scan_time - datetime.now(pytz.timezone('US/Eastern'))
                total_seconds = delta.total_seconds()
                if total_seconds > 0:
                    mins, secs = int(total_seconds // 60), int(total_seconds % 60)
                    col_t2.metric("До следующего скана", f"{mins} мин {secs} сек")
                else:
                    col_t2.metric("До следующего скана", "Запуск...")
            else:
                col_t1.metric("Последний авто-скан", "Ожидание...")
                col_t2.metric("До следующего скана", "Неизвестно")

            st.subheader("📜 Последние логи")
            with st.container(height=300):
                for log in reversed(BOT_STATE["logs"][-20:]): st.text(log)
            st.divider()
    except: pass

    TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN"))
    ADMIN_ID = st.secrets.get("ADMIN_ID", os.environ.get("ADMIN_ID"))
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", os.environ.get("GITHUB_USERS_URL"))
except:
    import os
    TG_TOKEN = os.environ.get("TG_TOKEN")
    ADMIN_ID = os.environ.get("ADMIN_ID")
    GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL")

def log_ui(message):
    print(message)
    ts = datetime.now().strftime('%H:%M:%S')
    BOT_STATE["logs"].append(f"[{ts}] {message}")
    if len(BOT_STATE["logs"]) > 100: BOT_STATE["logs"] = BOT_STATE["logs"][-100:]

if not TG_TOKEN:
    log_ui("CRITICAL ERROR: TG_TOKEN not found!")
    if 'st' in globals(): st.error("CRITICAL ERROR: TG_TOKEN not found!")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# === НОВЫЕ ДЕФОЛТНЫЕ НАСТРОЙКИ ===
DEFAULT_SETTINGS = {
    "portfolio_size": 100000,   # NEW DEFAULT
    "risk_per_trade_pct": 0.5,  # NEW DEFAULT
    "min_rr": 1.5,              # DEFAULT
    "len_major": 200,
    "len_fast": 20,
    "len_slow": 40,
    "adx_len": 14,
    "adx_thresh": 20,
    "atr_len": 14,
    "max_atr_pct": 5.0,         # DEFAULT
    "auto_scan": True,          # CHANGED TO TRUE (Default ON)
    "scan_mode": "S&P 500",     # NEW DEFAULT
    "show_new_only": True       # NEW DEFAULT (Only New)
}

user_settings = {}
ABORT_SCAN_USERS = set()
USER_STATES = {}
SENT_SIGNALS_CACHE = {"date": None, "tickers": set()}

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        html = pd.read_html(response.text, header=0)
        df = html[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

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
    sig_line = m.ewm(span=sig, adjust=False).mean()
    return m, sig_line, m - sig_line

def calc_adx(df, l):
    h, lo, c = df['High'], df['Low'], df['Close']
    up, down = h - h.shift(1), lo.shift(1) - lo
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    alpha = 1.0 / l
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    p_dm_s = pd.Series(p_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    m_dm_s = pd.Series(m_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    p_di = 100 * (p_dm_s / tr_s)
    m_di = 100 * (m_dm_s / tr_s)
    dx = 100 * ((p_di - m_di).abs() / (p_di + m_di))
    return dx.ewm(alpha=alpha, adjust=False).mean(), p_di, m_di

# ==========================================
# 3. ЛОГИКА СТРАТЕГИИ
# ==========================================
def run_strategy_for_ticker(ticker, settings):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < settings['len_major']: return None

        df['SMA_Major'] = calc_sma(df['Close'], settings['len_major'])
        adx, pdi, mdi = calc_adx(df, settings['adx_len'])
        atr = calc_atr(df, settings['atr_len'])
        df['EMA_Fast'] = calc_ema(df['Close'], settings['len_fast'])
        df['EMA_Slow'] = calc_ema(df['Close'], settings['len_slow'])
        _, _, macd_hist = calc_macd(df['Close'], 12, 26, 9)
        df['EFI'] = calc_ema(df['Close'].diff() * df['Volume'], settings['len_fast'])

        c_arr, h_arr, l_arr = df['Close'].values, df['High'].values, df['Low'].values
        ema_f, ema_s = df['EMA_Fast'].values, df['EMA_Slow'].values
        hist_vals, efi_vals = macd_hist.values, df['EFI'].values
        adx_v, pdi_v, mdi_v = adx.values, pdi.values, mdi.values

        n = len(df)
        t_list, s_list, crit_list, peak_list, struct_list = [0]*n, [0]*n, [np.nan]*n, [np.nan]*n, [False]*n

        seq_st, crit, s_h, s_l = 0, np.nan, h_arr[0], l_arr[0]
        l_peak, l_trough, l_hh, l_hl = np.nan, np.nan, False, False

        for i in range(1, n):
            c, h, l = c_arr[i], h_arr[i], l_arr[i]
            is_break = False
            if seq_st == 1 and not np.isnan(crit): is_break = c < crit
            elif seq_st == -1 and not np.isnan(crit): is_break = c > crit
            
            if is_break:
                if seq_st == 1:
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
            
            curr_st = 0
            if strong and (pdi_v[i] > mdi_v[i]) and rising and (hist_vals[i] > hist_vals[i-1]) and (efi_vals[i] > 0): curr_st = 1
            elif strong and (mdi_v[i] > pdi_v[i]) and falling and (hist_vals[i] < hist_vals[i-1]) and (efi_vals[i] < 0): curr_st = -1
            
            t_list[i], s_list[i], crit_list[i], peak_list[i], struct_list[i] = curr_st, seq_st, crit, l_peak, (l_hh and l_hl)

        def check(idx):
            if idx < 0: return False, 0.0, np.nan, np.nan
            p, sma = c_arr[idx], df['SMA_Major'].iloc[idx]
            valid = (s_list[idx] == 1) and ((p > sma) if not np.isnan(sma) else False) and (t_list[idx] != -1) and struct_list[idx]
            rr, cr, pk = 0.0, crit_list[idx], peak_list[idx]
            if valid and not np.isnan(pk) and not np.isnan(cr):
                rsk, rwd = p - cr, pk - p
                if rsk > 0 and rwd > 0: rr = rwd / rsk
                else: valid = False
            return valid, rr, cr, pk

        v_today, rr_t, sl_t, tp_t = check(n-1)
        v_yest, _, _, _ = check(n-2)
        
        if not v_today or rr_t < settings['min_rr']: return None
        
        cur_atr = atr.iloc[-1]
        atr_pct = (cur_atr / c_arr[-1]) * 100
        if atr_pct > settings['max_atr_pct']: return None
        
        risk_val = c_arr[-1] - sl_t
        risk_amt = settings['portfolio_size'] * (settings['risk_per_trade_pct'] / 100.0)
        shares = int(risk_amt / risk_val) if risk_val > 0 else 0
        shares = min(shares, int(settings['portfolio_size'] / c_arr[-1]))
        if shares < 1: shares = 1

        return {
            "Ticker": ticker, "Price": c_arr[-1], "RR": rr_t, "SL": sl_t, "TP": tp_t,
            "ATR_SL": c_arr[-1] - cur_atr, "Shares": shares, "ATR_Pct": atr_pct, 
            "Is_New": (v_today and not v_yest)
        }
    except: return None

# ==========================================
# 4. БОТ: ЛОГИКА
# ==========================================

async def check_auth_async(user_id):
    if ADMIN_ID and str(user_id) == str(ADMIN_ID): return True
    if not GITHUB_USERS_URL: return False
    try:
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(None, requests.get, GITHUB_USERS_URL)
        return (str(user_id) in [l.strip() for l in r.text.splitlines() if l.strip()]) if r.status_code == 200 else False
    except: return False

def get_settings(user_id):
    if user_id not in user_settings: user_settings[user_id] = DEFAULT_SETTINGS.copy()
    return user_settings[user_id]

def get_main_keyboard(user_id):
    s = get_settings(user_id)
    return ReplyKeyboardMarkup([
        [KeyboardButton("🚀 Запустить Скан")],
        [KeyboardButton("⚙️ Настройки"), KeyboardButton(f"🔄 Авто: {'✅' if s['auto_scan'] else '❌'}")]
    ], resize_keyboard=True)

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await check_auth_async(uid):
        await update.message.reply_text(f"⛔ Доступ запрещен. ID: `{uid}`")
        return
    await update.message.reply_text("👋 **Vova Screener Bot**\nМеню внизу 👇", reply_markup=get_main_keyboard(uid), parse_mode=ParseMode.MARKDOWN)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id
    if not await check_auth_async(uid): return

    # Обработка ввода значений
    if uid in USER_STATES:
        state = USER_STATES[uid]
        # Если нажата кнопка меню вместо ввода числа - отменяем ввод
        if text in ["🚀 Запустить Скан", "⚙️ Настройки"] or text.startswith("🔄 Авто:"):
            del USER_STATES[uid]
            await update.message.reply_text("Ввод отменен.", reply_markup=get_main_keyboard(uid))
        else:
            try:
                # Очистка ввода от % и $ и замена запятой
                clean_text = text.replace(',', '.').replace('%', '').replace('$', '').strip()
                val = float(clean_text)
                s = get_settings(uid)
                
                if state == "RISK": 
                    s['risk_per_trade_pct'] = val
                    await update.message.reply_text(f"✅ Risk обновлен: {val}%")
                elif state == "RR": 
                    s['min_rr'] = val
                    await update.message.reply_text(f"✅ Min RR обновлен: {val}")
                elif state == "PORT": 
                    s['portfolio_size'] = int(val)
                    await update.message.reply_text(f"✅ Portfolio обновлен: ${int(val)}")
                elif state == "ATR":
                    s['max_atr_pct'] = val
                    await update.message.reply_text(f"✅ Max ATR обновлен: {val}%")
                
                del USER_STATES[uid]
                await settings_menu(update, context) # Возвращаем меню настроек
                return
            except ValueError:
                await update.message.reply_text("❌ Введите корректное число (например 1.5).")
                return

    # Обработка команд меню
    if text == "🚀 Запустить Скан": await run_scan_task(update, context, uid, manual=True)
    elif text == "⚙️ Настройки": await settings_menu(update, context)
    elif text.startswith("🔄 Авто:"):
        s = get_settings(uid)
        s['auto_scan'] = not s['auto_scan']
        await update.message.reply_text(f"🔄 Авто-скан: {'ВКЛЮЧЕН' if s['auto_scan'] else 'ВЫКЛЮЧЕН'}", reply_markup=get_main_keyboard(uid))
    else:
        # Если пользователь ввел число, но не нажимал кнопку редактирования
        try:
            float(text.replace(',', '.').replace('%', '').replace('$', '').strip())
            await update.message.reply_text("⚠️ Чтобы изменить настройку, сначала нажмите соответствующую кнопку в меню '⚙️ Настройки'.", reply_markup=get_main_keyboard(uid))
        except:
            await update.message.reply_text("Выберите действие:", reply_markup=get_main_keyboard(uid))

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if not update.callback_query else update.callback_query.from_user.id
    msg_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
    
    s = get_settings(uid)
    txt = (
        f"⚙️ **Настройки:**\n"
        f"💰 Portfolio: ${s['portfolio_size']:,} | ⚠️ Risk: {s['risk_per_trade_pct']}%\n"
        f"📊 RR: {s['min_rr']} | 🔍 Mode: {s['scan_mode']}\n"
        f"📈 Max ATR: {s['max_atr_pct']}%\n"
        f"👀 Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'}"
    )
    kb = [
        [InlineKeyboardButton(f"Risk: {s['risk_per_trade_pct']}% ✏️", callback_data="ask_risk"),
         InlineKeyboardButton(f"RR: {s['min_rr']} ✏️", callback_data="ask_rr")],
        [InlineKeyboardButton(f"Portfolio: ${s['portfolio_size']} ✏️", callback_data="ask_port")],
        [InlineKeyboardButton(f"Max ATR: {s['max_atr_pct']}% ✏️", callback_data="ask_atr")],
        [InlineKeyboardButton(f"Mode: {s['scan_mode']} 🔄", callback_data="change_mode")],
        [InlineKeyboardButton(f"Фильтр: {'🔥 Только новые' if s.get('show_new_only', False) else '✅ Все активные'} 🔄", callback_data="toggle_filter")]
    ]
    await msg_func(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    s = get_settings(uid)
    d = query.data
    
    if d == "abort_scan":
        ABORT_SCAN_USERS.add(uid)
        await query.message.reply_text("🛑 Сканирование остановлено.")
        return

    if d == "ask_risk":
        USER_STATES[uid] = "RISK"
        await query.message.reply_text(f"Введите **Risk %** (сейчас: {s['risk_per_trade_pct']}):", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_rr":
        USER_STATES[uid] = "RR"
        await query.message.reply_text(f"Введите **Min RR** (сейчас: {s['min_rr']}):", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_port":
        USER_STATES[uid] = "PORT"
        await query.message.reply_text(f"Введите **Portfolio $** (сейчас: {s['portfolio_size']}):", parse_mode=ParseMode.MARKDOWN)
    elif d == "ask_atr":
        USER_STATES[uid] = "ATR"
        await query.message.reply_text(f"Введите **Max ATR %** (сейчас: {s['max_atr_pct']}):", parse_mode=ParseMode.MARKDOWN)
    elif d == "change_mode":
        s['scan_mode'] = "S&P 500" if s['scan_mode'] == "Top 10" else "Top 10"
        await settings_menu(update, context)
    elif d == "toggle_filter":
        s['show_new_only'] = not s.get('show_new_only', False)
        await settings_menu(update, context)

async def run_scan_task(update, context, uid, manual=False):
    s = get_settings(uid)
    msg_dest = update.message if update.message else update.callback_query.message
    filter_txt = "🔥 Только новые" if s.get('show_new_only', False) else "✅ Все активные"
    
    if uid in ABORT_SCAN_USERS: ABORT_SCAN_USERS.remove(uid)
    tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
    total = len(tickers)
    
    pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 СТОП", callback_data="abort_scan")]])
    status = await msg_dest.reply_text(f"🚀 Скан: {total} шт\nРежим: {filter_txt}", reply_markup=pkb)
    
    loop = asyncio.get_running_loop()
    found = 0
    batch = 5
    
    for i in range(0, total, batch):
        if uid in ABORT_SCAN_USERS:
            await status.edit_text(f"🛑 Прервано на {i}/{total}.")
            ABORT_SCAN_USERS.remove(uid)
            return
            
        for t in tickers[i:i+batch]:
            if uid in ABORT_SCAN_USERS: break
            res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
            if res:
                is_pass = res['Is_New'] if s.get('show_new_only', False) else True
                if is_pass:
                    found += 1
                    await send_signal_msg(context, uid, res)
        
        pct = int((i+len(tickers[i:i+batch]))/total*100)
        filled = int(10 * pct / 100)
        bar = "█"*filled + "░"*(10-filled)
        try: await status.edit_text(f"🚀 Скан: {pct}%\n[{bar}] {i+len(tickers[i:i+batch])}/{total}\nНайдено: {found}", reply_markup=pkb)
        except: pass

    try: await status.edit_text(f"✅ Готово! Найдено: {found}", reply_markup=None)
    except: await msg_dest.reply_text(f"✅ Готово! Найдено: {found}")

async def send_signal_msg(context, uid, res):
    tv_t = res['Ticker'].replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_t}"
    icon = "🔥 NEW" if res['Is_New'] else "✅ ACTIVE"
    
    msg = (
        f"{icon} **[{tv_t}]({tv_link})** | ${res['Price']:.2f}\n"
        f"📊 **ATR:** {res['ATR_Pct']:.2f}% | **ATR SL:** ${res['ATR_SL']:.2f}\n"
        f"🎯 **RR:** {res['RR']:.2f} | 🛑 **SL:** ${res['SL']:.2f}\n"
        f"🏁 **TP:** ${res['TP']:.2f} | 📦 **Size:** {res['Shares']} stocks"
    )
    await context.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    BOT_STATE["last_scan"] = now
    
    today = now.strftime("%Y-%m-%d")
    if SENT_SIGNALS_CACHE["date"] != today:
        SENT_SIGNALS_CACHE["date"] = today
        SENT_SIGNALS_CACHE["tickers"] = set()
    
    if now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0):
        log_ui(f"🔄 Auto-Scan Start... {now.strftime('%H:%M')}")
        for uid, s in user_settings.items():
            if s.get('auto_scan', False):
                tickers = get_top_10_tickers() if s['scan_mode'] == "Top 10" else get_sp500_tickers()
                loop = asyncio.get_running_loop()
                for t in tickers:
                    res = await loop.run_in_executor(None, run_strategy_for_ticker, t, s)
                    if res and res['Is_New'] and res['Ticker'] not in SENT_SIGNALS_CACHE["tickers"]:
                         await send_signal_msg(context, uid, res)
                         SENT_SIGNALS_CACHE["tickers"].add(res['Ticker'])
    else:
        log_ui(f"💤 Market Closed {now.strftime('%H:%M')}")

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
    def log_message(self, format, *args): return

def start_keep_alive():
    try:
        s = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
        threading.Thread(target=s.serve_forever, daemon=True).start()
    except: pass

if __name__ == '__main__':
    start_keep_alive()
    if TG_TOKEN:
        try:
            log_ui("Bot Init...")
            app = ApplicationBuilder().token(TG_TOKEN).build()
            app.add_handler(CommandHandler('start', start_handler))
            app.add_handler(CallbackQueryHandler(button_handler))
            app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
            app.job_queue.run_repeating(auto_scan_job, interval=3600, first=10)
            log_ui("Polling Started...")
            app.run_polling(stop_signals=[], drop_pending_updates=False)
        except Exception as e:
            log_ui(f"ERR: {e}")
    else: log_ui("No Token")
