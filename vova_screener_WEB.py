import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io

# Настройка страницы (чтобы выглядело как приложение)
st.set_page_config(page_title="Vova Screener", layout="wide")

st.title("S&P 500 Vova Screener")
st.caption("Поиск акций с 3 зелеными сигналами: MA, Sequence, Trend")

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
with st.sidebar:
    st.header("Настройки")
    LENGTH_MAJOR = st.number_input("SMA Major", value=200)
    ADX_LEN = st.number_input("ADX Length", value=14)
    ADX_THRESH = st.number_input("ADX Threshold", value=20)
    
    st.divider()
    
    # Выбор режима
    limit_tickers = st.slider("Сколько тикеров проверить (для теста)?", 10, 500, 50)
    show_all = st.checkbox("Показывать все (даже красные)?", value=False)

# ==========================================
# 2. ФУНКЦИИ (КЭШИРОВАНИЕ ДЛЯ СКОРОСТИ)
# ==========================================

@st.cache_data(ttl=86400) # Кэшируем список на 24 часа
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        table = pd.read_html(io.StringIO(response.text))
        tickers = table[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        st.error(f"Ошибка загрузки списка: {e}")
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOGL", "AMZN", "META", "BRK-B", "JPM"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_vova_status(df):
    if len(df) < 250: return None
    
    # 1. SMA
    df['SMA_Major'] = df['Close'].rolling(window=LENGTH_MAJOR).mean()
    
    # 2. ADX
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['Up'] = df['High'] - df['High'].shift(1)
    df['Down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
    df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
    
    tr_smooth = pine_rma(df['TR'], ADX_LEN)
    plus_dm_smooth = pine_rma(df['+DM'], ADX_LEN)
    minus_dm_smooth = pine_rma(df['-DM'], ADX_LEN)
    
    df['DI_Plus'] = 100 * (plus_dm_smooth / tr_smooth)
    df['DI_Minus'] = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
    df['ADX'] = pine_rma(dx, ADX_LEN)
    
    # 3. SEQUENCE LOGIC
    # Упрощаем логику для скорости (Vectorized logic is hard for loops, keeping manual loop for accuracy)
    seq_states = []
    seqState = 0
    seqHigh = df['High'].iloc[0]
    seqLow = df['Low'].iloc[0]
    criticalLevel = df['Low'].iloc[0]
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    for i in range(len(df)):
        if i == 0:
            seq_states.append(0)
            continue
        
        c, h, l = closes[i], highs[i], lows[i]
        prevSeqState = seq_states[-1]
        isBreak = False
        
        if prevSeqState == 1: isBreak = c < criticalLevel
        elif prevSeqState == -1: isBreak = c > criticalLevel
        
        if isBreak:
            if prevSeqState == 1:
                seqState = -1; seqHigh = h; seqLow = l; criticalLevel = h
            else:
                seqState = 1; seqHigh = h; seqLow = l; criticalLevel = l
        else:
            if seqState == 1:
                if h >= seqHigh: seqHigh = h
                if h >= seqHigh: criticalLevel = l
            elif seqState == -1:
                if l <= seqLow: seqLow = l
                if l <= seqLow: criticalLevel = h
            else:
                if c > seqHigh: seqState = 1; criticalLevel = l
                elif c < seqLow: seqState = -1; criticalLevel = h
                else: seqHigh = max(seqHigh, h); seqLow = min(seqLow, l)
        
        seq_states.append(seqState)
    
    last = df.iloc[-1]
    if pd.isna(last['ADX']) or pd.isna(last['SMA_Major']): return None
    
    is_seq_green = seq_states[-1] == 1
    is_ma_green = last['Close'] > last['SMA_Major']
    has_mom = last['ADX'] >= ADX_THRESH
    mom_bull = last['DI_Plus'] > last['DI_Minus']
    is_trend_green = has_mom and is_seq_green and mom_bull
    
    return {
        'Price': last['Close'],
        'Seq': "🟢" if is_seq_green else "🔴",
        'MA': "🟢" if is_ma_green else "🔴",
        'Trend': "🟢" if is_trend_green else "🔴",
        'All_Green': is_seq_green and is_ma_green and is_trend_green
    }

# ==========================================
# 3. ИНТЕРФЕЙС
# ==========================================

if st.button("🚀 ЗАПУСТИТЬ СКАНЕР"):
    tickers = get_sp500_tickers()[:limit_tickers]
    
    # Прогресс бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Сканирую {ticker} ({i+1}/{len(tickers)})...")
        try:
            # Скачиваем чуть меньше данных для скорости
            data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            if not data.empty:
                res = calculate_vova_status(data)
                if res:
                    res['Ticker'] = ticker
                    if res['All_Green'] or show_all:
                        results.append(res)
        except Exception:
            pass
        
        progress_bar.progress((i + 1) / len(tickers))
    
    status_text.text("Сканирование завершено!")
    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        # Красивая таблица
        st.subheader(f"Найдено: {len(df_res)}")
        
        # Переставляем колонки
        cols = ['Ticker', 'Price', 'All_Green', 'Seq', 'MA', 'Trend']
        df_res = df_res[cols]
        
        # Подсветка зеленых строк
        def highlight_green(row):
            return ['background-color: rgba(144, 238, 144, 0.2)'] * len(row) if row['All_Green'] else [''] * len(row)

        st.dataframe(
            df_res.style.apply(highlight_green, axis=1),
            use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "All_Green": st.column_config.CheckboxColumn(disabled=True)
            }
        )
    else:
        st.warning("Ничего не найдено с заданными параметрами.")