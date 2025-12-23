import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time
import math
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# ==========================================
# 1. КОНФИГУРАЦИЯ И СЕССИИ
# ==========================================
st.set_page_config(
    page_title="Vova Pro Screener", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Сессия для yfinance (защита от сбоев)
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session

YF_SESSION = get_session()

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def pine_rma(series, length):
    """Аналог ta.rma из Pine Script"""
    return series.ewm(alpha=1/length, adjust=False).mean()

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """Получает список S&P 500 с Википедии (кешируется на 1 час)"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        table = pd.read_html(io.StringIO(response.text))
        tickers = [str(t).replace('.', '-').strip() for t in table[0]['Symbol'].tolist()]
        return sorted(list(set(tickers)))
    except Exception as e:
        st.error(f"Ошибка загрузки тикеров: {e}")
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META"] # Fallback

# ==========================================
# 3. ЯДРО ЛОГИКИ (STRICT PINE MATCH)
# ==========================================
def check_ticker(ticker, settings):
    """
    Анализирует один тикер строго по алгоритму Pine Script 'sequence Vova (Super Trend)'
    """
    try:
        # 1. Загрузка данных (2 года для корректного расчета структуры)
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, session=YF_SESSION, timeout=10)
        
        # Обработка мультииндекса (pandas 2.0+ / yfinance update)
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass # Если структура простая, пропускаем

        if df.empty or len(df) < 250: 
            return None

        # =========================================================
        # A. РАСЧЕТ ИНДИКАТОРОВ
        # =========================================================
        
        # 1. SMA 200
        df['SMA_Major'] = df['Close'].rolling(window=settings["LENGTH_MAJOR"]).mean()
        
        # 2. ADX & DMI (Pine Logic)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        
        tr_smooth = pine_rma(df['TR'], 14)
        plus_dm_smooth = pine_rma(df['+DM'], 14)
        minus_dm_smooth = pine_rma(df['-DM'], 14)
        
        # Избегаем деления на 0
        tr_smooth = tr_smooth.replace(0, np.nan)
        
        df['DI_Plus'] = 100 * (plus_dm_smooth / tr_smooth)
        df['DI_Minus'] = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = pine_rma(dx, 14)

        # 3. Elder Impulse Components
        df['EMA_Fast'] = df['Close'].ewm(span=settings["LEN_FAST"], adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=settings["LEN_SLOW"], adjust=False).mean()
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd_line - signal_line
        
        # 4. Elder Force Index (EFI)
        df['Price_Change'] = df['Close'].diff()
        df['Raw_Force'] = df['Price_Change'] * df['Volume']
        df['EFI'] = df['Raw_Force'].ewm(span=settings["LEN_FAST"], adjust=False).mean()

        # =========================================================
        # B. ЛОГИКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ (ЦИКЛ)
        # =========================================================
        
        cl = df['Close'].values; hi = df['High'].values; lo = df['Low'].values
        
        seq_states = []
        hist_struct_valid = [] 
        hist_crit_level = []   
        hist_last_peak = []    

        # Инициализация переменных состояния
        seqState = 0 
        seqHigh = hi[0]
        seqLow = lo[0]
        crit = lo[0]
        
        # Память структуры
        lastConfirmedPeak = np.nan
        lastConfirmedTrough = np.nan
        lastPeakWasHH = False
        lastTroughWasHL = False
        
        length = len(df)
        
        for i in range(length):
            if i == 0:
                seq_states.append(0)
                hist_struct_valid.append(False)
                hist_crit_level.append(crit)
                hist_last_peak.append(np.nan)
                continue
            
            c, h, l = cl[i], hi[i], lo[i]
            pS = seq_states[-1]
            
            # Проверка пробоя уровня
            isBreak = False
            if pS == 1 and c < crit: isBreak = True
            elif pS == -1 and c > crit: isBreak = True
            
            if isBreak:
                if pS == 1: 
                    # >>> UP TREND BROKEN (Bearish) <<<
                    # Фаза 1: Был ли пик HH?
                    isCurrentPeakHH = False
                    if not np.isnan(lastConfirmedPeak):
                        if seqHigh > lastConfirmedPeak: isCurrentPeakHH = True
                    else:
                        isCurrentPeakHH = True # Первый пик считаем валидным
                    
                    # Фаза 2: Обновляем память
                    lastPeakWasHH = isCurrentPeakHH
                    lastConfirmedPeak = seqHigh
                    
                    # Сброс в Short
                    seqState = -1; seqHigh = h; seqLow = l; crit = h
                else: 
                    # >>> DOWN TREND BROKEN (Bullish) <<<
                    # Фаза 1: Был ли минимум HL?
                    isCurrentTroughHL = False
                    if not np.isnan(lastConfirmedTrough):
                        if seqLow > lastConfirmedTrough: isCurrentTroughHL = True # HL
                        elif seqLow < lastConfirmedTrough: isCurrentTroughHL = False # LL
                        else: isCurrentTroughHL = True # DB
                    else:
                        isCurrentTroughHL = True
                    
                    # Фаза 2: Обновляем память
                    lastTroughWasHL = isCurrentTroughHL
                    lastConfirmedTrough = seqLow
                    
                    # Сброс в Long
                    seqState = 1; seqHigh = h; seqLow = l; crit = l
            else:
                seqState = pS
                if seqState == 1:
                    if h >= seqHigh:
                        seqHigh = h
                        crit = l # Trail Stop (Low)
                    # Если новый хай не перебит, стоп не двигаем
                elif seqState == -1:
                    if l <= seqLow:
                        seqLow = l
                        crit = h
                else:
                    # Инициализация
                    if c > seqHigh: seqState = 1; crit = l
                    elif c < seqLow: seqState = -1; crit = h
                    else:
                        seqHigh = max(seqHigh, h)
                        seqLow = min(seqLow, l)
            
            seq_states.append(seqState)
            hist_struct_valid.append(lastPeakWasHH and lastTroughWasHL)
            hist_crit_level.append(crit)
            hist_last_peak.append(lastConfirmedPeak)

        # =========================================================
        # C. ФИНАЛЬНАЯ ПРОВЕРКА (ПОСЛЕДНИЙ БАР)
        # =========================================================
        idx = -1 
        
        # 1. SEQUENCE STATE
        cond_seq = (seq_states[idx] == 1)
        
        # 2. MA STATE
        cond_ma = (df['Close'].iloc[idx] > df['SMA_Major'].iloc[idx])
        
        # 3. SUPER TREND STATE
        adx = df['ADX'].iloc[idx]
        di_plus = df['DI_Plus'].iloc[idx]
        di_minus = df['DI_Minus'].iloc[idx]
        
        adx_bull = (adx >= settings["ADX_THRESH"]) and (di_plus > di_minus)
        adx_bear = (adx >= settings["ADX_THRESH"]) and (di_minus > di_plus)
        
        # Impulse: Сравнение текущего и предыдущего
        ema_f_curr = df['EMA_Fast'].iloc[idx]; ema_f_prev = df['EMA_Fast'].iloc[idx-1]
        ema_s_curr = df['EMA_Slow'].iloc[idx]; ema_s_prev = df['EMA_Slow'].iloc[idx-1]
        hist_curr = df['MACD_Hist'].iloc[idx]; hist_prev = df['MACD_Hist'].iloc[idx-1]
        
        # Строгий бычий импульс: ВСЕ три должны расти
        elder_bull = (ema_f_curr > ema_f_prev) and (ema_s_curr > ema_s_prev) and (hist_curr > hist_prev)
        elder_bear = (ema_f_curr < ema_f_prev) and (ema_s_curr < ema_s_prev) and (hist_curr < hist_prev)
        
        efi = df['EFI'].iloc[idx]
        efi_bull = efi > 0
        efi_bear = efi < 0
        
        # Собираем статус тренда
        trend_state_val = 0
        if adx_bull and elder_bull and efi_bull:
            trend_state_val = 1
        elif adx_bear and elder_bear and efi_bear:
            trend_state_val = -1
        
        # Условие входа: Тренд НЕ должен быть красным
        cond_trend = (trend_state_val != -1)
        
        # 4. STRUCTURE VALIDITY
        cond_struct = hist_struct_valid[idx]
        
        # КОМБИНАЦИЯ ВСЕХ УСЛОВИЙ
        is_setup_valid = cond_seq and cond_ma and cond_trend and cond_struct
        
        # =========================================================
        # D. РАСЧЕТ RR И РИСКА
        # =========================================================
        rr_val = 0.0
        shares = 0
        dollar_risk_val = 0.0
        risk_per_share = 0.0
        
        current_crit = hist_crit_level[idx]
        current_target = hist_last_peak[idx] # TP = Последний подтвержденный пик
        close_price = df['Close'].iloc[idx]
        
        if is_setup_valid and not np.isnan(current_crit) and not np.isnan(current_target):
            risk_per_share = close_price - current_crit
            reward_per_share = current_target - close_price
            
            if risk_per_share > 0 and reward_per_share > 0:
                rr_val = reward_per_share / risk_per_share
                
                # Расчет позиции
                dollar_risk_val = settings["PORTFOLIO_SIZE"] * (settings["RISK_PER_TRADE"] / 100.0)
                shares = math.floor(dollar_risk_val / risk_per_share)
            else:
                rr_val = -999.0 # Цена выше цели или ниже стопа
        
        # ФИЛЬТРЫ
        has_setup = (rr_val > 0)
        pass_rr = (rr_val >= settings["MIN_RR"])
        
        # ATR Check
        atr_val = df['TR'].tail(14).mean()
        atr_pct = (atr_val / close_price) * 100
        pass_atr = (atr_pct <= settings["MAX_ATR_PCT"])
        
        # New Signal Check (для метки "NEW")
        is_new_signal = (seq_states[idx-1] != 1) or (not hist_struct_valid[idx-1])

        if has_setup and pass_rr and pass_atr:
            return {
                'Ticker': ticker,
                'Price': round(close_price, 2),
                'RR': round(rr_val, 2),
                'TP': round(current_target, 2),
                'SL': round(current_crit, 2),
                'Risk ($)': round(dollar_risk_val, 0),
                'Shares': shares,
                'ATR %': round(atr_pct, 2),
                'Is New': is_new_signal
            }
            
    except Exception as e:
        return None
    return None

# ==========================================
# 4. ИНТЕРФЕЙС STREAMLIT
# ==========================================

# --- САЙДБАР: НАСТРОЙКИ ---
st.sidebar.header("⚙️ Настройки Стратегии")

# Параметры индикатора (Строгие)
st.sidebar.subheader("Индикаторы (Strict)")
sma_len = st.sidebar.number_input("Major SMA Period", value=200, step=10)
ema_fast = st.sidebar.number_input("Impulse Fast EMA", value=20, step=1)
ema_slow = st.sidebar.number_input("Impulse Slow EMA", value=40, step=1)
adx_thresh = st.sidebar.number_input("ADX Threshold", value=20, step=1)

# Параметры Риска
st.sidebar.subheader("Риск Менеджмент")
portfolio = st.sidebar.number_input("Размер портфеля ($)", value=100000.0, step=1000.0)
risk_pct = st.sidebar.number_input("Риск на сделку (%)", value=0.5, step=0.1)
min_rr = st.sidebar.number_input("Мин. Risk/Reward", value=1.5, step=0.1)
max_atr = st.sidebar.number_input("Макс. ATR (%)", value=5.0, step=0.5)

# Режим отображения
st.sidebar.subheader("Фильтры отображения")
show_only_new = st.sidebar.checkbox("Только НОВЫЕ сигналы", value=False)

# Сборка настроек в словарь
SETTINGS = {
    "LENGTH_MAJOR": sma_len,
    "LEN_FAST": ema_fast,
    "LEN_SLOW": ema_slow,
    "ADX_THRESH": adx_thresh,
    "PORTFOLIO_SIZE": portfolio,
    "RISK_PER_TRADE": risk_pct,
    "MIN_RR": min_rr,
    "MAX_ATR_PCT": max_atr
}

# --- ОСНОВНОЙ ЭКРАН ---
st.title("🦅 Vova Pro Screener")
st.markdown(f"**Стратегия:** Strict Super Trend + Structure HH/HL + Risk Management")
st.markdown(f"**Портфель:** ${portfolio:,.0f} | **Риск:** {risk_pct}% | **Цель RR:** > {min_rr}")

# Кнопка запуска
if st.button("🚀 ЗАПУСТИТЬ СКАНЕР S&P 500", type="primary"):
    tickers = get_sp500_tickers()
    st.info(f"Загружено тикеров: {len(tickers)}. Начинаю анализ...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Многопоточный запуск для скорости
    # Используем ThreadPoolExecutor, так как основная задержка - это I/O (сеть)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Создаем словарь {future: ticker}
        future_to_ticker = {executor.submit(check_ticker, t, SETTINGS): t for t in tickers}
        
        completed = 0
        total = len(tickers)
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                res = future.result()
                if res:
                    # Фильтр "Только Новые" применяется здесь, перед добавлением в список
                    if not show_only_new or res['Is New']:
                        results.append(res)
            except Exception as exc:
                pass
            
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Обработано: {completed}/{total} ({int(progress*100)}%)")

    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.success(f"✅ Сканирование завершено! Найдено сигналов: {len(results)}")
        
        # Создаем DataFrame для красивого вывода
        df_res = pd.DataFrame(results)
        
        # Сортировка: Сначала New, потом по RR (от большего к меньшему)
        df_res = df_res.sort_values(by=['Is New', 'RR'], ascending=[False, False])
        
        # Форматирование и вывод
        st.dataframe(
            df_res,
            column_config={
                "Is New": st.column_config.CheckboxColumn("New?", disabled=True),
                "RR": st.column_config.NumberColumn("Risk/Reward", format="%.2f"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "TP": st.column_config.NumberColumn("Take Profit", format="$%.2f"),
                "SL": st.column_config.NumberColumn("Stop Loss", format="$%.2f"),
                "Risk ($)": st.column_config.NumberColumn("Risk $", format="$%d"),
                "ATR %": st.column_config.NumberColumn("ATR", format="%.2f%%"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Экспорт в CSV
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать CSV",
            csv,
            "vova_scan_results.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("🏁 Сигналов не найдено. Попробуйте смягчить фильтры (например, RR).")

else:
    st.info("Нажмите кнопку выше, чтобы начать поиск.")import streamlit as st
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
import json
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Bot Server", page_icon="🤖", layout="centered")

# ==========================================
# 1. НАСТРОЙКИ И АВТОРИЗАЦИЯ
# ==========================================

try:
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
    ADMIN_ID = int(st.secrets.get("ADMIN_ID", 0))
    TG_TOKEN = st.secrets["TG_TOKEN"]
except:
    GITHUB_USERS_URL = ""
    ADMIN_ID = 0
    TG_TOKEN = ""

if not TG_TOKEN:
    st.error("❌ Токен не найден в Secrets!")
    st.stop()

# Сессия для yfinance (защита от блокировок Yahoo Finance)
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

YF_SESSION = get_session()

def fetch_approved_ids():
    """Загружает список ID из GitHub Raw URL"""
    ids = set()
    if ADMIN_ID != 0: ids.add(ADMIN_ID)
    if not GITHUB_USERS_URL: return ids
    try:
        response = requests.get(GITHUB_USERS_URL, timeout=10)
        if response.status_code == 200:
            for line in response.text.splitlines():
                line = line.strip()
                if line.isdigit(): ids.add(int(line))
    except Exception as e: print(f"GitHub fetch error: {e}")
    return ids

bot = telebot.TeleBot(TG_TOKEN, threaded=False)

@st.cache_resource
def get_shared_state():
    return {
        "LENGTH_MAJOR": 200, 
        "LEN_FAST": 20,      # EMA 20 (Impulse & EFI)
        "LEN_SLOW": 40,      # EMA 40 (Impulse)
        "MAX_ATR_PCT": 5.0, 
        "ADX_THRESH": 20,    
        "MIN_RR": 1.5,       
        "AUTO_SCAN_INTERVAL": 3600, 
        "IS_SCANNING": False, 
        "STOP_SCAN": False,
        "SHOW_ONLY_NEW": False, # CHANGED: По умолчанию False, чтобы видеть ВСЕ сигналы сразу
        "LAST_SCAN_TIME": "Никогда",
        "CHAT_IDS": set(), 
        "APPROVED_IDS": fetch_approved_ids(), 
        "NOTIFIED_TODAY": set(), 
        "LAST_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "TIMEZONE_OFFSET": -7.0 
    }

SETTINGS = get_shared_state()

# ГЛОБАЛЬНЫЙ ПРОГРЕСС ДЛЯ BUFFER BAR
PROGRESS = {
    "current": 0, "total": 0, "running": False, "msg_id": None, "chat_id": None, "header": ""
}

def is_authorized(user_id):
    if ADMIN_ID != 0 and user_id == ADMIN_ID: return True
    return user_id in SETTINGS["APPROVED_IDS"]

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2, one_time_keyboard=False)
    markup.row(types.KeyboardButton('Scan 🚀'), types.KeyboardButton('Stop 🛑'))
    markup.row(types.KeyboardButton('Status 📊'), types.KeyboardButton('Mode 🔄'))
    markup.row(types.KeyboardButton('ATR 📉'), types.KeyboardButton('RR 🎯'), types.KeyboardButton('Time 🕒'))
    return markup

def get_local_now():
    return datetime.utcnow() + timedelta(hours=SETTINGS["TIMEZONE_OFFSET"])

# ==========================================
# 2. ФУНКЦИИ АНАЛИЗА (TRADING VIEW LOGIC)
# ==========================================
def get_sp500_tickers():
    for attempt in range(3):
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            table = pd.read_html(io.StringIO(response.text))
            tickers = [str(t).replace('.', '-').strip() for t in table[0]['Symbol'].tolist()]
            return sorted(list(set(tickers)))
        except:
            time.sleep(2)
            if attempt == 2: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    """
    ПОЛНАЯ СИНХРОНИЗАЦИЯ С PINE SCRIPT (SUPER TREND + TRADE LOGIC)
    """
    try:
        # 1. Загрузка данных (2 года)
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True, session=YF_SESSION, timeout=10)
        if df.empty or len(df) < 250: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

        # =========================================================
        # 2. РАСЧЕТ ИНДИКАТОРОВ (ELDER + ADX + MA)
        # =========================================================
        
        # A. SMA 200
        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        # B. ADX & DMI
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        
        tr_smooth = pine_rma(df['TR'], 14)
        plus_dm_smooth = pine_rma(df['+DM'], 14)
        minus_dm_smooth = pine_rma(df['-DM'], 14)
        
        df['DI_Plus'] = 100 * (plus_dm_smooth / tr_smooth)
        df['DI_Minus'] = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = pine_rma(dx, 14)

        # C. ELDER IMPULSE COMPONENTS
        # EMA 20 & EMA 40
        df['EMA_Fast'] = df['Close'].ewm(span=SETTINGS["LEN_FAST"], adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=SETTINGS["LEN_SLOW"], adjust=False).mean()
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd_line - signal_line
        
        # D. ELDER FORCE INDEX (EFI)
        # EMA 20 of (Change * Volume)
        df['Price_Change'] = df['Close'].diff()
        df['Raw_Force'] = df['Price_Change'] * df['Volume']
        df['EFI'] = df['Raw_Force'].ewm(span=SETTINGS["LEN_FAST"], adjust=False).mean()

        # =========================================================
        # 3. VOVA SEQUENCE LOGIC (STRUCTURE MEMORY)
        # =========================================================
        # Используем весь DF, а не tail, чтобы максимизировать историю для поиска пиков
        df_calc = df.copy() 
        if len(df_calc) < 5: return None
        
        cl = df_calc['Close'].values; hi = df_calc['High'].values; lo = df_calc['Low'].values
        
        seq_states = []
        seqState = 0 
        seqHigh = hi[0]
        seqLow = lo[0]
        crit = lo[0]
        
        # Память структуры
        lastConfirmedPeak = np.nan
        lastConfirmedTrough = np.nan
        lastPeakWasHH = False
        lastTroughWasHL = False
        
        # Вспомогательные списки для истории
        hist_struct_valid = [] 
        hist_crit_level = []   
        hist_last_peak = []    

        for i in range(len(df_calc)):
            if i == 0:
                seq_states.append(0)
                hist_struct_valid.append(False)
                hist_crit_level.append(crit)
                hist_last_peak.append(np.nan)
                continue
            
            c, h, l = cl[i], hi[i], lo[i]
            pS = seq_states[-1]
            
            # Пробой уровня
            isBreak = (pS == 1 and c < crit) or (pS == -1 and c > crit)
            
            if isBreak:
                if pS == 1: # Long -> Short
                    # Был UP, стал DOWN. Текущий seqHigh фиксируется как Пик.
                    if not np.isnan(lastConfirmedPeak):
                        if seqHigh > lastConfirmedPeak: lastPeakWasHH = True
                        else: lastPeakWasHH = False
                    else:
                        lastPeakWasHH = True # Первый пик считаем HH для старта
                    
                    lastConfirmedPeak = seqHigh # Запоминаем уровень пика
                    
                    seqState = -1; seqHigh = h; seqLow = l; crit = h
                else: # Short -> Long
                    # Был DOWN, стал UP. Текущий seqLow фиксируется как Дно.
                    if not np.isnan(lastConfirmedTrough):
                        if seqLow > lastConfirmedTrough: lastTroughWasHL = True
                        elif seqLow < lastConfirmedTrough: lastTroughWasHL = False
                        else: lastTroughWasHL = True # DB
                    else:
                        lastTroughWasHL = True
                    
                    lastConfirmedTrough = seqLow
                    
                    seqState = 1; seqHigh = h; seqLow = l; crit = l
            else:
                seqState = pS
                if seqState == 1:
                    if h >= seqHigh:
                        seqHigh = h
                        crit = l # Trail Stop
                    else:
                        pass # crit remains same
                elif seqState == -1:
                    if l <= seqLow:
                        seqLow = l
                        crit = h
                else:
                    if c > seqHigh: seqState = 1; crit = l
                    elif c < seqLow: seqState = -1; crit = h
                    else:
                        seqHigh = max(seqHigh, h)
                        seqLow = min(seqLow, l)
            
            seq_states.append(seqState)
            hist_struct_valid.append(lastPeakWasHH and lastTroughWasHL)
            hist_crit_level.append(crit)
            hist_last_peak.append(lastConfirmedPeak)

        # =========================================================
        # 4. ФИНАЛЬНАЯ ПРОВЕРКА (SUPER TREND + TRADE LOGIC)
        # =========================================================
        
        idx_cur = -1
        idx_prev = -2
        
        row_cur = df_calc.iloc[idx_cur]
        row_prev = df_calc.iloc[idx_prev]
        
        def check_conditions(row, idx_offset):
            # 1. Sequence Green
            cond_seq = (seq_states[idx_offset] == 1)
            
            # 2. MA Green
            cond_ma = (row['Close'] > row['SMA_Major'])
            
            # 3. Super Trend (Strict Green)
            adx_bull = (row['ADX'] >= SETTINGS["ADX_THRESH"]) and (row['DI_Plus'] > row['DI_Minus'])
            
            # Elder Impulse (Double EMA Rising + Hist Rising)
            pos = len(df_calc) + idx_offset
            if pos < 1: return False
            
            curr_ema_f = df_calc['EMA_Fast'].iloc[pos]
            prev_ema_f = df_calc['EMA_Fast'].iloc[pos-1]
            curr_ema_s = df_calc['EMA_Slow'].iloc[pos]
            prev_ema_s = df_calc['EMA_Slow'].iloc[pos-1]
            curr_hist = df_calc['MACD_Hist'].iloc[pos]
            prev_hist = df_calc['MACD_Hist'].iloc[pos-1]
            
            elder_bull = (curr_ema_f > prev_ema_f) and (curr_ema_s > prev_ema_s) and (curr_hist > prev_hist)
            
            # EFI
            efi_bull = row['EFI'] > 0
            
            cond_trend = (adx_bull and elder_bull and efi_bull)
            
            # 4. Structure Valid
            cond_struct = hist_struct_valid[idx_offset]
            
            return (cond_seq and cond_ma and cond_trend and cond_struct)

        is_setup_valid = check_conditions(row_cur, idx_cur)
        was_setup_valid = check_conditions(row_prev, idx_prev)
        is_new_signal = is_setup_valid and not was_setup_valid
        
        # --- ATR FILTER ---
        atr_val = df['TR'].tail(14).mean()
        atr_pct = (atr_val / row_cur['Close']) * 100
        pass_atr = (atr_pct <= SETTINGS["MAX_ATR_PCT"])
        
        # --- RR CALCULATION ---
        rr_val = 0.0
        current_crit = hist_crit_level[idx_cur]
        current_target = hist_last_peak[idx_cur] # TP = Previous Confirmed Peak
        
        if is_setup_valid and not np.isnan(current_crit) and not np.isnan(current_target):
            risk = row_cur['Close'] - current_crit
            reward = current_target - row_cur['Close']
            
            if risk > 0 and reward > 0:
                rr_val = reward / risk
        
        pass_rr = (rr_val >= SETTINGS["MIN_RR"])

        if is_setup_valid and pass_atr and pass_rr:
            if not SETTINGS["SHOW_ONLY_NEW"] or is_new_signal:
                return {
                    'ticker': ticker, 
                    'price': row_cur['Close'], 
                    'atr': atr_pct, 
                    'rr': rr_val,
                    'is_new': is_new_signal
                }
    except Exception as e:
        return None
    return None

# ==========================================
# 3. УПРАВЛЕНИЕ СКАНЕРОМ И BUFFER BAR
# ==========================================

def progress_updater():
    """Фоновое обновление сообщения с Buffer Bar в Telegram"""
    while PROGRESS["running"]:
        try:
            if PROGRESS["total"] > 0:
                pct = int((PROGRESS["current"] / PROGRESS["total"]) * 100)
                bar_str = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
                text = (f"{PROGRESS['header']}\n"
                        f"SMA: {SETTINGS['LENGTH_MAJOR']} | RR > {SETTINGS['MIN_RR']}\n"
                        f"Прогресс: {PROGRESS['current']}/{PROGRESS['total']} ({pct}%)\n"
                        f"[{bar_str}]")
                bot.edit_message_text(chat_id=PROGRESS["chat_id"], message_id=PROGRESS["msg_id"], text=text, parse_mode="HTML")
        except: pass
        time.sleep(5)

def perform_scan(chat_id, is_manual=False):
    if SETTINGS["IS_SCANNING"]:
        try: bot.send_message(chat_id, "⚠️ Сканирование уже идет!")
        except: pass
        return
    
    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    try:
        local_now = get_local_now()
        current_date_str = local_now.strftime("%Y-%m-%d")
        if SETTINGS["LAST_DATE"] != current_date_str:
            SETTINGS["NOTIFIED_TODAY"] = set()
            SETTINGS["LAST_DATE"] = current_date_str
        
        header = "🚀 <b>Ручной поиск</b>" if is_manual else "⏰ <b>Авто-проверка</b>"
        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        start_msg = bot.send_message(chat_id, f"{header}\n⏳ Подготовка данных...", parse_mode="HTML")
        
        PROGRESS.update({
            "current": 0, "total": total_tickers, "running": True, 
            "msg_id": start_msg.message_id, "chat_id": chat_id, "header": header
        })
        
        threading.Thread(target=progress_updater, daemon=True).start()
        
        found_count = 0
        for i, t in enumerate(tickers):
            if SETTINGS["STOP_SCAN"]: 
                PROGRESS["running"] = False
                bot.send_message(chat_id, "🛑 Сканирование остановлено пользователем.")
                break
            
            PROGRESS["current"] = i + 1
            if i > 0 and i % 20 == 0: time.sleep(0.5)
            
            res = check_ticker(t)
            if res:
                if not is_manual and res['ticker'] in SETTINGS["NOTIFIED_TODAY"]: 
                    continue
                
                if not is_manual: SETTINGS["NOTIFIED_TODAY"].add(res['ticker'])
                
                found_count += 1
                icon = "🔥 NEW" if res['is_new'] else "🟢"
                msg = (f"{icon} <b>{res['ticker']}</b>\n"
                       f"Price: ${res['price']:.2f}\n"
                       f"RR: <b>{res['rr']:.2f}</b> (Min: {SETTINGS['MIN_RR']})\n"
                       f"ATR: {res['atr']:.2f}%")
                
                targets = [chat_id] if is_manual else list(SETTINGS["CHAT_IDS"])
                for target in targets:
                    if is_authorized(target):
                        try: bot.send_message(target, msg, parse_mode="HTML")
                        except: pass
        
        PROGRESS["running"] = False
        final_text = f"✅ <b>Завершено</b>. Найдено сигналов: {found_count}" if found_count > 0 else "🏁 <b>Завершено</b>. Подходящих акций не найдено."
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=start_msg.message_id, text=final_text, parse_mode="HTML")
        except:
            bot.send_message(chat_id, final_text, parse_mode="HTML")
            
    except Exception as e:
        PROGRESS["running"] = False
        bot.send_message(chat_id, f"❌ Ошибка сервера: {str(e)}")
    finally:
        SETTINGS["IS_SCANNING"] = False
        SETTINGS["LAST_SCAN_TIME"] = get_local_now().strftime("%H:%M:%S")

# ==========================================
# 4. ОБРАБОТЧИКИ ТЕЛЕГРАМ
# ==========================================

@bot.message_handler(func=lambda m: not is_authorized(m.from_user.id))
def unauthorized_access(message):
    bot.send_message(message.chat.id, 
        f"⛔ <b>Доступ ограничен.</b>\nID: <code>{message.from_user.id}</code>\nСвяжитесь с админом.", parse_mode="HTML")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    SETTINGS["CHAT_IDS"].add(message.chat.id)
    bot.send_message(message.chat.id, "👋 <b>Vova Bot Server Active</b>\nS&P 500 Scanner + RR Filter.", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['reload'])
def reload_users(message):
    if message.from_user.id != ADMIN_ID: return
    SETTINGS["APPROVED_IDS"] = fetch_approved_ids()
    bot.send_message(ADMIN_ID, f"✅ Список ID обновлен ({len(SETTINGS['APPROVED_IDS'])} чел.)")

@bot.message_handler(func=lambda m: m.text == 'Scan 🚀')
def manual_scan(message):
    SETTINGS["CHAT_IDS"].add(message.chat.id)
    threading.Thread(target=perform_scan, args=(message.chat.id, True)).start()

@bot.message_handler(func=lambda m: m.text == 'Stop 🛑')
def stop_scan(message):
    SETTINGS["STOP_SCAN"] = True
    bot.reply_to(message, "🛑 Команда остановки принята.")

@bot.message_handler(func=lambda m: m.text == 'Status 📊')
def get_status(message):
    mode = "Только НОВЫЕ" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ активные"
    bot.reply_to(message, f"⚙️ <b>Статус:</b>\nРежим: {mode}\nОдобрено: {len(SETTINGS['APPROVED_IDS'])}\nSMA: {SETTINGS['LENGTH_MAJOR']}\nRR Min: {SETTINGS['MIN_RR']}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nПосл. скан: {SETTINGS['LAST_SCAN_TIME']}", parse_mode="HTML")

@bot.message_handler(func=lambda m: m.text == 'Mode 🔄')
def open_mode_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    markup.add('Только НОВЫЕ 🔥', 'ВСЕ активные 🟢', '🔙 Назад')
    bot.send_message(message.chat.id, "🔄 <b>Выберите режим отображения:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == 'Только НОВЫЕ 🔥')
def set_mode_new(message):
    SETTINGS["SHOW_ONLY_NEW"] = True
    bot.reply_to(message, "✅ Режим: <b>Только НОВЫЕ сигналы</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == 'ВСЕ активные 🟢')
def set_mode_all(message):
    SETTINGS["SHOW_ONLY_NEW"] = False
    bot.reply_to(message, "✅ Режим: <b>ВСЕ активные сигналы</b>", parse_mode="HTML", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == '🔙 Назад')
def back_to_main(message):
    bot.send_message(message.chat.id, "🏠 Главное меню", reply_markup=get_main_keyboard())

# --- ATR MENU ---
@bot.message_handler(func=lambda m: m.text == 'ATR 📉')
def open_atr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add('3.0 %', '5.0 %', '7.0 %', '10.0 %', '🔙 Назад')
    bot.send_message(message.chat.id, "📉 <b>Выберите Max ATR %:</b>", parse_mode="HTML", reply_markup=markup)

# --- SMA MENU ---
@bot.message_handler(func=lambda m: m.text == 'SMA 📈')
def open_sma_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add('100', '150', '200', '🔙 Назад')
    bot.send_message(message.chat.id, "📈 <b>Выберите SMA Period:</b>", parse_mode="HTML", reply_markup=markup)

# --- RR MENU (NEW) ---
@bot.message_handler(func=lambda m: m.text == 'RR 🎯')
def open_rr_menu(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add('RR 1.0', 'RR 1.5', 'RR 2.0', 'RR 2.5', 'RR 3.0', '🔙 Назад')
    bot.send_message(message.chat.id, "🎯 <b>Выберите Min Risk/Reward:</b>", parse_mode="HTML", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text.startswith('RR '))
def set_rr_value(message):
    try:
        val = float(message.text.split(' ')[1])
        SETTINGS["MIN_RR"] = val
        bot.reply_to(message, f"✅ Min RR установлен: <b>{val}</b>", parse_mode="HTML", reply_markup=get_main_keyboard())
    except: pass

@bot.message_handler(func=lambda m: '%' in m.text or m.text.isdigit())
def handle_values(message):
    if '%' in message.text:
        try:
            val = float(message.text.replace(' %',''))
            SETTINGS["MAX_ATR_PCT"] = val
            bot.reply_to(message, f"✅ Max ATR: {val}%", reply_markup=get_main_keyboard())
        except: pass
    elif message.text.isdigit():
        try:
            val = int(message.text)
            SETTINGS["LENGTH_MAJOR"] = val
            bot.reply_to(message, f"✅ SMA Period: {val}", reply_markup=get_main_keyboard())
        except: pass

@bot.message_handler(func=lambda m: m.text == 'Time 🕒')
def check_time(message):
    local_time = get_local_now().strftime("%H:%M")
    bot.reply_to(message, f"🕒 Ваше время (UTC-7): <b>{local_time}</b>", parse_mode="HTML")

# ==========================================
# 5. СЕРВИСЫ
# ==========================================
def start_polling():
    while True:
        try: bot.infinity_polling(timeout=20)
        except: time.sleep(5)

def start_scheduler():
    while True:
        time.sleep(3600) # Авто-скан раз в час
        for chat_id in list(SETTINGS["CHAT_IDS"]):
            if is_authorized(chat_id):
                perform_scan(chat_id, False)

@st.cache_resource
def run_background_services():
    threading.Thread(target=start_polling, daemon=True).start()
    threading.Thread(target=start_scheduler, daemon=True).start()
    return True

# ==========================================
# 6. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.title("🤖 Vova Bot Server")
run_background_services()
st.success(f"✅ Сервер активен. Одобрено пользователей (GitHub): {len(SETTINGS['APPROVED_IDS'])}")
st.metric("Последний скан", SETTINGS["LAST_SCAN_TIME"])
st.metric("S&P 500 Tickers", "Loading..." if not 'tickers' in st.session_state else len(st.session_state['tickers']))

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300000, key="ref")

