import streamlit as st
import telebot
from telebot import types
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading
import requests
import os
from datetime import datetime, timedelta
from threading import Lock

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ======================================================
# TELEGRAM TOKEN
# ======================================================
TG_TOKEN = st.secrets.get("TG_TOKEN", os.environ.get("TG_TOKEN", ""))
if not TG_TOKEN:
    st.error("Telegram token not found")
    st.stop()

bot = telebot.TeleBot(TG_TOKEN, threaded=False)

# ======================================================
# SHARED SETTINGS
# ======================================================
@st.cache_resource
def shared_state():
    return {
        "LENGTH_MAJOR": 200,
        "MAX_ATR_PCT": 5.0,
        "ADX_THRESH": 20,
        "SHOW_ONLY_NEW": True,
        "TICKER_LIMIT": 500,
        "TIMEZONE_OFFSET": -7,
        "NOTIFIED_TODAY": set(),
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "CHAT_ID": None,
        "STOP_SCAN": False,
        "IS_SCANNING": False
    }

SETTINGS = shared_state()

# ======================================================
# PROGRESS STATE (THREAD SAFE)
# ======================================================
PROGRESS = {
    "running": False,
    "current": 0,
    "total": 0,
    "percent": 0,
    "found": 0,
    "last_ticker": "",
    "started_at": None
}

PROGRESS_LOCK = Lock()

# ======================================================
# UTILS
# ======================================================
def get_local_now():
    return datetime.utcnow() + timedelta(hours=SETTINGS["TIMEZONE_OFFSET"])

def get_main_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("Scan 🚀", "Stop 🛑")
    kb.row("Status 📊", "Mode 🔄")
    kb.row("ATR 📉", "SMA 📈", "Limit 🔢")
    return kb

# ======================================================
# DATA
# ======================================================
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(io.StringIO(requests.get(url).text))[0]
        return [t.replace(".", "-") for t in table["Symbol"].tolist()]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

# ======================================================
# TICKER CHECK
# ======================================================
def check_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if len(df) < 250:
            return None

        df["SMA"] = df["Close"].rolling(SETTINGS["LENGTH_MAJOR"]).mean()
        df["TR"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1))
            )
        )
        df["ATR"] = pine_rma(df["TR"], 14)
        df["ATR_PCT"] = df["ATR"] / df["Close"] * 100

        up = df["High"] - df["High"].shift(1)
        down = df["Low"].shift(1) - df["Low"]
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)

        tr = pine_rma(df["TR"], 14)
        plus_di = 100 * pine_rma(pd.Series(plus_dm), 14) / tr
        minus_di = 100 * pine_rma(pd.Series(minus_dm), 14) / tr
        adx = pine_rma(100 * abs(plus_di - minus_di) / (plus_di + minus_di), 14)

        last = df.iloc[-1]
        if np.isnan(adx.iloc[-1]):
            return None

        conditions = (
            last["Close"] > last["SMA"] and
            adx.iloc[-1] >= SETTINGS["ADX_THRESH"] and
            plus_di.iloc[-1] > minus_di.iloc[-1] and
            last["ATR_PCT"] <= SETTINGS["MAX_ATR_PCT"]
        )

        if conditions:
            return {
                "ticker": ticker,
                "price": last["Close"],
                "atr": last["ATR_PCT"]
            }
    except:
        return None
    return None

# ======================================================
# SCAN
# ======================================================
def perform_scan(chat_id):
    if SETTINGS["IS_SCANNING"]:
        return

    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False

    tickers = get_sp500_tickers()[:SETTINGS["TICKER_LIMIT"]]

    with PROGRESS_LOCK:
        PROGRESS.update({
            "running": True,
            "current": 0,
            "total": len(tickers),
            "percent": 0,
            "found": 0,
            "last_ticker": "",
            "started_at": time.time()
        })

    for i, t in enumerate(tickers):
        if SETTINGS["STOP_SCAN"]:
            break

        res = check_ticker(t)

        with PROGRESS_LOCK:
            PROGRESS["current"] = i + 1
            PROGRESS["percent"] = int((i + 1) / len(tickers) * 100)
            PROGRESS["last_ticker"] = t
            if res:
                PROGRESS["found"] += 1

        if res:
            bot.send_message(
                chat_id,
                f"🟢 {res['ticker']} | ${res['price']:.2f} | ATR {res['atr']:.2f}%",
                reply_markup=get_main_keyboard()
            )

    with PROGRESS_LOCK:
        PROGRESS["running"] = False

    SETTINGS["IS_SCANNING"] = False

# ======================================================
# TELEGRAM PROGRESS UPDATER
# ======================================================
def telegram_progress(chat_id):
    msg = bot.send_message(chat_id, "⏳ Scan started")
    last = -1

    while True:
        time.sleep(5)
        with PROGRESS_LOCK:
            if not PROGRESS["running"]:
                break
            p = PROGRESS["percent"]
            c = PROGRESS["current"]
            t = PROGRESS["total"]
            f = PROGRESS["found"]
            l = PROGRESS["last_ticker"]

        if p != last:
            try:
                bot.edit_message_text(
                    f"⏳ Progress: {c}/{t} ({p}%)\nLast: {l}\nFound: {f}",
                    chat_id,
                    msg.message_id
                )
            except:
                pass
            last = p

# ======================================================
# TELEGRAM HANDLERS
# ======================================================
@bot.message_handler(commands=["start"])
def start(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    bot.send_message(message.chat.id, "Bot ready", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == "Scan 🚀")
def scan(message):
    SETTINGS["CHAT_ID"] = message.chat.id
    threading.Thread(target=perform_scan, args=(message.chat.id,), daemon=True).start()
    threading.Thread(target=telegram_progress, args=(message.chat.id,), daemon=True).start()

@bot.message_handler(func=lambda m: m.text == "Stop 🛑")
def stop(message):
    SETTINGS["STOP_SCAN"] = True
    bot.reply_to(message, "Stopping…")

# ======================================================
# SERVICES
# ======================================================
def polling():
    while True:
        try:
            bot.infinity_polling(timeout=20)
        except:
            time.sleep(5)

@st.cache_resource
def run_services():
    threading.Thread(target=polling, daemon=True).start()
    return True

# ======================================================
# STREAMLIT UI
# ======================================================
st.title("🤖 Vova Bot Server")

run_services()

st.subheader("📊 Live Progress")

with PROGRESS_LOCK:
    running = PROGRESS["running"]
    percent = PROGRESS["percent"]
    current = PROGRESS["current"]
    total = PROGRESS["total"]
    found = PROGRESS["found"]
    last = PROGRESS["last_ticker"]

st.progress(percent / 100 if running else 0)
st.write(f"Running: {running}")
st.write(f"{current}/{total}")
st.write(f"Found: {found}")
st.write(f"Last ticker: {last}")

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=2000, key="refresh")
