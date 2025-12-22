import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Vova Screener", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 12px; height: 3em; font-weight: bold;}
    [data-testid="stExpander"] { border-radius: 12px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("📱 Vova Mobile Screener")

# --- Инициализация состояния для защиты от повторных алертов ---
if 'sent_alerts' not in st.session_state:
    st.session_state['sent_alerts'] = set()

# ==========================================
# 2. НАСТРОЙКИ ФИЛЬТРОВ И УВЕДОМЛЕНИЙ
# ==========================================
with st.expander("⚙️ Настройки и Уведомления", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Индикаторы")
        LENGTH_MAJOR = st.number_input("SMA Period", value=200)
        MAX_ATR_PCT = st.number_input("Max ATR %", value=10.0, step=0.5)
        MIN_MCAP = st.number_input("Min Market Cap (B$)", value=10.0, step=5.0)

    with col2:
        st.subheader("🔔 Telegram Уведомления")
        tg_token = st.text_input("Bot Token", placeholder="123456:ABC-DEF...", type="password", help="Получить у @BotFather")
        tg_chat_id = st.text_input("Chat ID", placeholder="12345678", help="Ваш ID или ID канала")
        check_interval = st.number_input("Интервал проверки (мин)", value=60, min_value=15, step=15)
        
        # --- КНОПКА ПРОВЕРКИ ---
        if st.button("📨 Проверить связь (Тест)"):
            if tg_token and tg_chat_id:
                try:
                    url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                    payload = {"chat_id": tg_chat_id, "text": "👋 <b>Тест прошел успешно!</b>\nБот готов к работе.", "parse_mode": "HTML"}
                    res = requests.post(url, json=payload)
                    if res.status_code == 200:
                        st.success("✅ Сообщение отправлено! Проверьте Telegram.")
                    else:
                        st.error(f"❌ Ошибка Telegram: {res.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка сети: {e}")
            else:
                st.warning("⚠️ Сначала введите Token и Chat ID!")

    st.divider()
    
    # --- ВЫБОР РЕЖИМА ПОИСКА ---
    search_mode = st.radio("Источник тикеров:", ["S&P 500 (Авто)", "Свой список (Вручную)"], horizontal=True)
    
    custom_tickers = []
    limit_tickers = 50 # Значение по умолчанию
    
    if search_mode == "S&P 500 (Авто)":
        limit_tickers = st.slider("Лимит тикеров для сканирования", 10, 503, 50)
    else:
        manual_input = st.text_area("Введите тикеры (через пробел или запятую)", "AAPL TSLA NVDA BTC-USD")
        if manual_input:
            # Очистка и форматирование списка
            custom_tickers = [t.strip().upper() for t in manual_input.replace(',', ' ').split() if t.strip()]
            st.caption(f"Будет проверено тикеров: {len(custom_tickers)}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        show_all = st.checkbox("Показывать все (даже красные)", value=False)
    with c2:
        show_only_new = st.checkbox("🔥 Только НОВЫЕ (вход сегодня)", value=False)

# ==========================================
# 3. ЛОГИКА
# ==========================================

def send_telegram(message, token, chat_id):
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
    except:
        pass

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))
        tickers = table[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "JPM", "NFLX"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_data(df, ticker_symbol):
    if len(df) < 250: return None
    
    # 1. SMA
    df['SMA_Major'] = df['Close'].rolling(window=LENGTH_MAJOR).mean()
    
    # 2. ATR Calculation
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['ATR_Val'] = df['TR'].rolling(window=14).mean()
    df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
    
    # 3. ADX
    ADX_THRESH_INTERNAL = 20 
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
    
    # 4. SEQUENCE
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
    
    # Market Cap
    try:
        t_info = yf.Ticker(ticker_symbol).fast_info
        mcap_billions = t_info.market_cap / 1_000_000_000
    except:
        mcap_billions = 0.0

    last = df.iloc[-1]
    if pd.isna(last['ADX']): return None

    # Current State
    seq_cur = seq_states[-1] == 1
    ma_cur = last['Close'] > last['SMA_Major']
    mom_cur = (last['ADX'] >= ADX_THRESH_INTERNAL) and seq_cur and (last['DI_Plus'] > last['DI_Minus'])
    all_green_cur = seq_cur and ma_cur and mom_cur

    # Previous State (for New Signal logic)
    prev = df.iloc[-2]
    seq_prev = seq_states[-2] == 1
    ma_prev = prev['Close'] > prev['SMA_Major']
    mom_prev = (prev['ADX'] >= ADX_THRESH_INTERNAL) and seq_prev and (prev['DI_Plus'] > prev['DI_Minus'])
    all_green_prev = seq_prev and ma_prev and mom_prev
    
    is_new_signal = all_green_cur and not all_green_prev

    pass_atr = last['ATR_Pct'] <= MAX_ATR_PCT 
    pass_mcap = mcap_billions >= MIN_MCAP

    return {
        'Ticker': ticker_symbol,
        'Price': last['Close'],
        'ATR_Pct': last['ATR_Pct'],
        'MCap_B': mcap_billions,
        'Seq': "🟢" if seq_cur else "🔴",
        'MA': "🟢" if ma_cur else "🔴",
        'Trend': "🟢" if mom_cur else "🔴",
        'All_Green': all_green_cur,
        'Is_New': is_new_signal,
        'Pass_Filters': pass_atr and pass_mcap
    }

# ==========================================
# 4. ИНТЕРФЕЙС И ЗАПУСК
# ==========================================

# Кнопки режимов
col_b1, col_b2 = st.columns(2)
start_manual = col_b1.button("🚀 ПОИСК (Один раз)", type="primary")
start_auto = col_b2.button("📡 АВТО-МОНИТОРИНГ (Loop)")

def run_scan(is_auto_mode=False):
    # Определение списка тикеров в зависимости от режима
    if search_mode == "S&P 500 (Авто)":
        tickers = get_sp500_tickers()[:limit_tickers]
    else:
        tickers = custom_tickers
        if not tickers:
            if not is_auto_mode:
                st.error("⚠️ Список тикеров пуст! Введите тикеры в настройках.")
            return []
    
    if is_auto_mode:
        placeholder_status = st.empty()
        placeholder_table = st.empty()
    else:
        my_bar = st.progress(0, text="Сканирование...")
    
    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            if not data.empty:
                res = calculate_data(data, ticker)
                if res and res['Pass_Filters']:
                    
                    # Логика для МОНИТОРИНГА
                    if is_auto_mode:
                        # Если сигнал НОВЫЙ и мы его еще не отправляли
                        if res['Is_New'] and ticker not in st.session_state['sent_alerts']:
                            msg = f"🚀 <b>NEW SIGNAL: {ticker}</b>\nPrice: ${res['Price']:.2f}\nATR: {res['ATR_Pct']:.2f}%"
                            send_telegram(msg, tg_token, tg_chat_id)
                            st.session_state['sent_alerts'].add(ticker)
                    
                    # Логика для ТАБЛИЦЫ
                    if show_only_new:
                        if res['Is_New']: results.append(res)
                    elif show_all or res['All_Green']:
                        results.append(res)
                            
        except Exception:
            pass
            
        if not is_auto_mode and i % 5 == 0:
            my_bar.progress((i + 1) / len(tickers), text=f"Scan: {ticker}")

    if not is_auto_mode:
        my_bar.empty()
        return results
    else:
        return results

# --- ЛОГИКА РУЧНОГО ЗАПУСКА ---
if start_manual:
    results = run_scan(is_auto_mode=False)
    if results:
        df_res = pd.DataFrame(results)
        if 'Is_New' in df_res.columns:
            df_res['Ticker'] = df_res.apply(lambda x: f"🔥 {x['Ticker']}" if x['Is_New'] else x['Ticker'], axis=1)
        st.success(f"Найдено: {len(df_res)}")
        st.dataframe(df_res, hide_index=True, use_container_width=True)
    else:
        st.warning("Ничего не найдено.")

# --- ЛОГИКА АВТО-МОНИТОРИНГА ---
if start_auto:
    st.info(f"🟢 Мониторинг запущен! Проверка каждые {check_interval} минут. Не закрывайте вкладку.")
    status_box = st.empty()
    result_box = st.empty()
    
    while True:
        current_time = time.strftime("%H:%M:%S")
        status_box.markdown(f"⏳ **Последняя проверка:** {current_time} | Сканирую рынок...")
        
        # Запускаем сканирование
        scan_results = run_scan(is_auto_mode=True)
        
        # Обновляем таблицу на экране (чтобы видно было, что происходит)
        if scan_results:
            df_auto = pd.DataFrame(scan_results)
            result_box.dataframe(df_auto, hide_index=True, use_container_width=True)
        
        status_box.success(f"✅ Проверка завершена в {current_time}. Следующая через {check_interval} мин.")
        
        # Ждем (Streamlit sleep)
        time.sleep(check_interval * 60)
        # Очистка кэша данных перед следующим циклом, чтобы подтянуть свежие цены
        get_sp500_tickers.clear() # Опционально
