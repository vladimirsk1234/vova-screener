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

# --- Инициализация состояния ---
if 'sent_alerts' not in st.session_state:
    st.session_state['sent_alerts'] = set()

# ==========================================
# 2. НАСТРОЙКИ
# ==========================================
with st.expander("⚙️ Настройки и Уведомления", expanded=True): # Открыто по умолчанию для настройки
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Индикаторы")
        LENGTH_MAJOR = st.number_input("SMA Period", value=200)
        MAX_ATR_PCT = st.number_input("Max ATR %", value=10.0, step=0.5)
        MIN_MCAP = st.number_input("Min Market Cap (B$)", value=10.0, step=5.0)

    with col2:
        st.subheader("🔔 Telegram (Обязательно нажмите Enter после ввода!)")
        tg_token = st.text_input("Bot Token", placeholder="123456:ABC-DEF...", type="password", help="Из @BotFather")
        tg_chat_id = st.text_input("Chat ID", placeholder="12345678", help="Из @userinfobot")
        check_interval = st.number_input("Интервал (мин)", value=60, min_value=15)
        
        # --- КНОПКА ПРОВЕРКИ ---
        if st.button("📨 Отправить тестовое сообщение"):
            if not tg_token or not tg_chat_id:
                st.error("⚠️ Вы не ввели Token или Chat ID! Заполните поля и нажмите Enter.")
            else:
                try:
                    url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                    payload = {"chat_id": tg_chat_id, "text": "👋 <b>Привет! Я работаю.</b>\nЕсли вы видите это, значит настройки верны.", "parse_mode": "HTML"}
                    res = requests.post(url, json=payload)
                    
                    if res.status_code == 200:
                        st.success("✅ УСПЕХ! Сообщение отправлено в Telegram.")
                        st.balloons()
                    elif res.status_code == 401:
                        st.error("❌ ОШИБКА 401: Неверный Bot Token. Проверьте, не скопировали ли вы лишние пробелы.")
                    elif res.status_code == 400:
                        st.error("❌ ОШИБКА 400: Неверный Chat ID. Или бот не запущен (нажмите /start в боте).")
                    else:
                        st.error(f"❌ Ошибка Telegram ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка сети: {e}")

    st.divider()
    
    search_mode = st.radio("Источник тикеров:", ["S&P 500 (Авто)", "Свой список (Вручную)"], horizontal=True)
    custom_tickers = []
    limit_tickers = 50
    
    if search_mode == "S&P 500 (Авто)":
        limit_tickers = st.slider("Лимит тикеров", 10, 503, 50)
    else:
        manual_input = st.text_area("Введите тикеры (AAPL TSLA ...)", "AAPL TSLA NVDA BTC-USD")
        if manual_input:
            custom_tickers = [t.strip().upper() for t in manual_input.replace(',', ' ').split() if t.strip()]

    c1, c2 = st.columns(2)
    with c1: show_all = st.checkbox("Показывать все", value=False)
    with c2: show_only_new = st.checkbox("🔥 Только новые сигналы", value=False)

# ==========================================
# 3. ЛОГИКА
# ==========================================

def send_telegram_alert(message, token, chat_id):
    if not token or not chat_id: return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})
    except: pass

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))
        return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "JPM"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_data(df, ticker_symbol):
    if len(df) < 250: return None
    
    # Tech Indicators
    df['SMA_Major'] = df['Close'].rolling(window=LENGTH_MAJOR).mean()
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR_Val'] = df['TR'].rolling(window=14).mean()
    df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
    
    # ADX
    df['Up'] = df['High'] - df['High'].shift(1)
    df['Down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
    df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
    tr_s = pine_rma(df['TR'], 14); p_dm = pine_rma(df['+DM'], 14); m_dm = pine_rma(df['-DM'], 14)
    df['DI_Plus'] = 100 * (p_dm / tr_s); df['DI_Minus'] = 100 * (m_dm / tr_s)
    df['ADX'] = pine_rma(100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus']), 14)
    
    # Sequence Logic
    seq_states = []
    seqState = 0; seqHigh = df['High'].iloc[0]; seqLow = df['Low'].iloc[0]; crit = df['Low'].iloc[0]
    cl = df['Close'].values; hi = df['High'].values; lo = df['Low'].values
    
    for i in range(len(df)):
        if i==0: seq_states.append(0); continue
        c, h, l = cl[i], hi[i], lo[i]
        pS = seq_states[-1]
        brk = (pS == 1 and c < crit) or (pS == -1 and c > crit)
        
        if brk:
            if pS == 1: seqState = -1; seqHigh = h; seqLow = l; crit = h
            else: seqState = 1; seqHigh = h; seqLow = l; crit = l
        else:
            if seqState == 1:
                if h >= seqHigh: seqHigh = h
                crit = l if h >= seqHigh else crit
            elif seqState == -1:
                if l <= seqLow: seqLow = l
                crit = h if l <= seqLow else crit
            else:
                if c > seqHigh: seqState = 1; crit = l
                elif c < seqLow: seqState = -1; crit = h
                else: seqHigh = max(seqHigh, h); seqLow = min(seqLow, l)
        seq_states.append(seqState)
    
    # Market Cap
    try: mcap = yf.Ticker(ticker_symbol).fast_info.market_cap / 1_000_000_000
    except: mcap = 0.0

    last = df.iloc[-1]; prev = df.iloc[-2]
    if pd.isna(last['ADX']): return None

    # Logic
    def get_status(row, seq):
        s = seq == 1
        m = row['Close'] > row['SMA_Major']
        t = (row['ADX'] >= 20) and s and (row['DI_Plus'] > row['DI_Minus'])
        return s, m, t, (s and m and t)

    s_c, m_c, t_c, ag_c = get_status(last, seq_states[-1])
    _, _, _, ag_p = get_status(prev, seq_states[-2])
    
    return {
        'Ticker': ticker_symbol, 'Price': last['Close'], 'ATR_Pct': last['ATR_Pct'], 'MCap_B': mcap,
        'Seq': "🟢" if s_c else "🔴", 'MA': "🟢" if m_c else "🔴", 'Trend': "🟢" if t_c else "🔴",
        'All_Green': ag_c, 'Is_New': ag_c and not ag_p,
        'Pass_Filters': (last['ATR_Pct'] <= MAX_ATR_PCT) and (mcap >= MIN_MCAP)
    }

# ==========================================
# 4. ЗАПУСК
# ==========================================
c_b1, c_b2 = st.columns(2)
start_manual = c_b1.button("🚀 ПОИСК (Вручную)", type="primary")
start_auto = c_b2.button("📡 АВТО-МОНИТОРИНГ")

def run_scan(is_auto=False):
    tickers = custom_tickers if search_mode == "Свой список (Вручную)" else get_sp500_tickers()[:limit_tickers]
    if not tickers and not is_auto: st.error("Пустой список тикеров!"); return []
    
    if is_auto:
        status_ph = st.empty()
    else:
        bar = st.progress(0, "Сканирование...")
    
    res_list = []
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if not df.empty:
                r = calculate_data(df, t)
                if r and r['Pass_Filters']:
                    # Auto Logic
                    if is_auto and r['Is_New'] and t not in st.session_state['sent_alerts']:
                        send_telegram_alert(f"🚀 <b>SIGNAL: {t}</b>\n${r['Price']:.2f}", tg_token, tg_chat_id)
                        st.session_state['sent_alerts'].add(t)
                    
                    # Table Logic
                    if show_only_new:
                        if r['Is_New']: res_list.append(r)
                    elif show_all or r['All_Green']:
                        res_list.append(r)
        except: pass
        
        if not is_auto and i % 5 == 0: bar.progress((i+1)/len(tickers))
        
    if not is_auto: bar.empty()
    return res_list

if start_manual:
    data = run_scan(False)
    if data:
        df = pd.DataFrame(data)
        if 'Is_New' in df.columns: df['Ticker'] = df.apply(lambda x: f"🔥 {x['Ticker']}" if x['Is_New'] else x['Ticker'], axis=1)
        st.success(f"Найдено: {len(df)}")
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Пусто.")

if start_auto:
    if not tg_token or not tg_chat_id:
        st.error("❌ Для мониторинга нужно ввести Token и Chat ID!")
        st.stop()
        
    st.toast("Мониторинг запущен!", icon="🟢")
    status = st.empty()
    table_ph = st.empty()
    
    while True:
        t_str = time.strftime("%H:%M:%S")
        status.info(f"⏳ {t_str}: Сканирование рынка...")
        data = run_scan(True)
        if data:
            table_ph.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
        
        status.success(f"✅ {t_str}: Готово. Жду {check_interval} мин.")
        time.sleep(check_interval * 60)
        get_sp500_tickers.clear()
