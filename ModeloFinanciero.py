import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import pandas_ta as ta
import statsmodels.api as sm

# ==========================================
# 1. CONFIGURACION Y UX (BLOOMBERG STYLE)
# ==========================================
st.set_page_config(page_title="QUANT TERMINAL MX", layout="wide", initial_sidebar_state="expanded")

# CSS para Terminal Bloomberg (Negro, Verde Neon y Ambar)
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00FF00; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #00FF00 !important; font-family: 'Courier New', Courier, monospace; }
    .stSidebar { background-color: #050505 !important; border-right: 1px solid #00FF00; }
    hr { border-color: #00FF00; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background-color: #000000 !important; color: #FFBF00 !important; border: 1px solid #00FF00 !important; }
    .stSelectbox>div>div { background-color: #000000 !important; color: #FFBF00 !important; border: 1px solid #00FF00 !important; }
    .stButton>button, [data-testid="baseButton-secondaryFormSubmit"] { background-color: #000000 !important; color: #FFBF00 !important; border: 1px solid #FFBF00 !important; border-radius: 0px !important; font-weight: bold; width: 100%; }
    .stButton>button:hover, [data-testid="baseButton-secondaryFormSubmit"]:hover { background-color: #FFBF00 !important; color: #000000 !important; }
    div[data-testid="metric-container"] { background-color: #0A0A0A; border: 1px solid #FFBF00; padding: 15px; border-radius: 0px; }
    div[data-testid="stMetricValue"] { color: #FFBF00 !important; font-size: 1.8rem !important; }
    div[data-testid="stMetricLabel"] { color: #00FF00 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTOR DE DATOS LOCAL (BMV / BIVA)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period="5y"):
    formatted_tickers = [t.strip().upper() + ".MX" if not t.upper().endswith(".MX") else t.strip().upper() for t in tickers.split(",")]
    if "^MXX" not in formatted_tickers:
        formatted_tickers.append("^MXX") 
    
    data = yf.download(formatted_tickers, period=period, progress=False)
    
    if data.empty:
        return None, None
        
    prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    prices = prices.ffill().bfill()
    
    low_liquidity = []
    if 'Volume' in data:
        volume = data['Volume'].ffill().bfill()
        for t in formatted_tickers:
            if t != "^MXX" and t in volume.columns:
                if volume[t].mean() < 10000:
                    low_liquidity.append(t)
                
    return prices, low_liquidity

# ==========================================
# 3. CORE ANALITICO & QUANTS
# ==========================================
def calc_beta_alpha(stock_returns, market_returns, rf_daily):
    y = stock_returns - rf_daily
    x = market_returns - rf_daily
    x = sm.add_constant(x)
    model = sm.OLS(y, x, missing='drop').fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    return beta, alpha * 252

def calc_altman_z(ticker_obj):
    try:
        bs = ticker_obj.balance_sheet
        ist = ticker_obj.income_stmt
        if not bs.empty and not ist.empty:
            A = (bs.loc['Total Current Assets'][0] - bs.loc['Total Current Liabilities'][0]) / bs.loc['Total Assets'][0]
            B = bs.loc['Retained Earnings'][0] / bs.loc['Total Assets'][0]
            C = ist.loc['EBIT'][0] / bs.loc['Total Assets'][0]
            D = ticker_obj.info.get('marketCap', 1) / bs.loc['Total Liabilities Net Minority Interest'][0]
            E = ist.loc['Total Revenue'][0] / bs.loc['Total Assets'][0]
            return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        return np.nan
    except: return np.nan

def montecarlo_sim(prices, days=252, sims=5000):
    returns = np.log(prices / prices.shift(1)).dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    paths = np.zeros((days, sims))
    paths[0] = prices.iloc[-1]
    
    for t in range(1, days):
        rand_shock = np.random.normal(0, 1, sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand_shock)
        
    return paths

def optimize_portfolio(returns, rf_annual):
    num_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    def neg_sharpe(weights):
        p_ret = np.sum(mean_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - rf_annual) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# ==========================================
# 4. ENRUTAMIENTO UI (SIDEBAR)
# ==========================================
st.sidebar.markdown("## SYS CMD")
st.sidebar.markdown("---")
with st.sidebar.form("macro_params"):
    rf_input = st.number_input("CETES 28D (Rf) %", value=11.00, step=0.1) / 100
    embi_input = st.number_input("EMBI Mexico (Riesgo Pais) %", value=3.50, step=0.1) / 100
    st.form_submit_button("UPDATE MACRO")

rf_total = rf_input + embi_input 
rf_daily = rf_total / 252

menu = st.sidebar.radio("COMMAND LINE:", ["[MARKET]", "[EQUITY]", "[RISK]", "[PORTFOLIO]", "[HELP/EDU]"])
st.sidebar.markdown("---")

# ==========================================
# 5. MODULOS DE LA TERMINAL
# ==========================================

if menu == "[MARKET]":
    st.markdown("## [MARKET] MACRO & BENCHMARK MX")
    st.markdown("---")
    if st.button("RUN MARKET DATA"):
        with st.spinner("FETCHING ^MXX..."):
            prices, low_liq = fetch_market_data("^MXX", "1y")
            if prices is not None:
                idx_data = prices['^MXX']
                ret_1y = (idx_data.iloc[-1] / idx_data.iloc[0]) - 1
                
                c1, c2, c3 = st.columns(3)
                c1.metric("S&P/BMV IPC Ultimo", f"{idx_data.iloc[-1]:,.2f}", f"{ret_1y:.2%} 1Y")
                c2.metric("Tasa Libre de Riesgo (CETES)", f"{rf_input*100:.2f}%")
                c3.metric("Riesgo Pais (EMBI+ MX)", f"{embi_input*100:.2f}%")
                
                fig = px.line(idx_data, title="Evolucion S&P/BMV IPC (^MXX)", color_discrete_sequence=["#00FF00"])
                fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00', xaxis_title="Fecha", yaxis_title="Puntos")
                fig.update_xaxes(showgrid=True, gridcolor='#333')
                fig.update_yaxes(showgrid=True, gridcolor='#333')
                st.plotly_chart(fig, use_container_width=True)

elif menu == "[EQUITY]":
    st.markdown("## [EQUITY] ANALISIS INDIVIDUAL (DES)")
    st.markdown("---")
    
    with st.form("eq_form"):
        ticker = st.text_input("TICKER (ej. WALMEX, FEMSAUBD):", "WALMEX")
        run_des = st.form_submit_button("RUN DES")
        
    if run_des:
        with st.spinner(f"ANALYZING {ticker}..."):
            prices, low_liq = fetch_market_data(ticker, "2y")
            if prices is not None:
                t_str = ticker.upper() + ".MX" if not ticker.upper().endswith(".MX") else ticker.upper()
                if t_str in low_liq:
                    st.warning(f"ADVERTENCIA: {t_str} presenta bajo volumen de operacion.")
                
                ret = np.log(prices / prices.shift(1)).dropna()
                beta, alpha = calc_beta_alpha(ret[t_str], ret["^MXX"], rf_daily)
                
                df_ta = yf.download(t_str, period="1y", progress=False)
                df_ta.ta.bbands(append=True)
                
                st.markdown("### FUNDAMENTALS & QUANTS")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Beta (Riesgo Sist.)", f"{beta:.2f}")
                c2.metric("Alfa de Jensen", f"{alpha:.2%}")
                
                capm_e = rf_total + beta * ((ret["^MXX"].mean()*252) - rf_total)
                c3.metric("Retorno CAPM (K_e)", f"{capm_e:.2%}")
                
                z_score = calc_altman_z(yf.Ticker(t_str))
                z_display = f"{z_score:.2f}" if not np.isnan(z_score) else "N/D"
                c4.metric("Altman Z-Score", z_display)
                
                st.markdown("### TECHNICAL ANALYSIS (BBANDS)")
                bb_upper = [col for col in df_ta.columns if col.startswith('BBU')][0]
                bb_lower = [col for col in df_ta.columns if col.startswith('BBL')][0]
                
                fig = go.Figure(data=[go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'], low=df_ta['Low'], close=df_ta['Close'], name="Precio")])
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta[bb_upper], line=dict(color='rgba(255,191,0,0.5)', dash='dash'), name="BB Upper"))
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta[bb_lower], line=dict(color='rgba(255,191,0,0.5)', dash='dash'), name="BB Lower"))
                fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00', xaxis_rangeslider_visible=False)
                fig.update_xaxes(showgrid=True, gridcolor='#333')
                fig.update_yaxes(showgrid=True, gridcolor='#333')
                st.plotly_chart(fig, use_container_width=True)

elif menu == "[RISK]":
    st.markdown("## [RISK] FORWARD-LOOKING & STRESS TEST")
    st.markdown("---")
    
    col_sim, col_tornado = st.columns(2)
    with col_sim:
        st.markdown("### SIMULACION MONTE CARLO (1 ANO)")
        with st.form("risk_form"):
            ticker = st.text_input("TICKER:", "ALFAA")
            run_mc = st.form_submit_button("RUN MONTE CARLO")
            
        if run_mc:
            with st.spinner("RUNNING SIMULATIONS..."):
                prices, _ = fetch_market_data(ticker, "3y")
                if prices is not None:
                    t_str = ticker.upper() + ".MX" if not ticker.upper().endswith(".MX") else ticker.upper()
                    paths = montecarlo_sim(prices[t_str], days=252, sims=5000)
                    
                    q5 = np.percentile(paths, 5, axis=1)
                    q50 = np.percentile(paths, 50, axis=1)
                    q95 = np.percentile(paths, 95, axis=1)
                    var_5 = (q5[-1] / paths[0][0]) - 1
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=q95, line=dict(color='rgba(0,255,0,0.2)'), name="P95 (Optimista)"))
                    fig.add_trace(go.Scatter(y=q5, line=dict(color='rgba(255,0,0,0.2)'), fill='tonexty', fillcolor='rgba(255,191,0,0.1)', name="P5 (Pesimista)"))
                    fig.add_trace(go.Scatter(y=q50, line=dict(color='#00FF00', width=2), name="P50 (Esperado)"))
                    fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00')
                    st.plotly_chart(fig, use_container_width=True)
                    st.error(f"VaR (95%): Maxima perdida esperada a 1 ano: {var_5:.2%}")

    with col_tornado:
        st.markdown("### SENSITIVITY (TORNADO)")
        inf = st.slider("Inflacion MX (%)", 2.0, 8.0, 4.0)
        g = st.slider("Crecimiento Terminal (g) (%)", 0.0, 5.0, 2.0)
        
        base_val = 100
        val_inf_up = base_val * (1 - (inf/100 * 1.5))
        val_inf_down = base_val * (1 + (inf/100 * 0.5))
        val_g_up = base_val * (1 + (g/100 * 2))
        val_g_down = base_val * (1 - (g/100 * 1))
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(y=['Inflacion', 'Crecimiento'], x=[val_inf_down - base_val, val_g_down - base_val], name='Impacto Negativo', orientation='h', marker_color='#FF0000'))
        fig_t.add_trace(go.Bar(y=['Inflacion', 'Crecimiento'], x=[val_inf_up - base_val, val_g_up - base_val], name='Impacto Positivo', orientation='h', marker_color='#00FF00'))
        fig_t.update_layout(barmode='relative', plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00')
        st.plotly_chart(fig_t, use_container_width=True)

elif menu == "[PORTFOLIO]":
    st.markdown("## [PORTFOLIO] MARKOWITZ & SWITCHING")
    st.markdown("---")
    
    with st.form("port_form"):
        port_tickers = st.text_input("TICKERS (coma):", "WALMEX, FEMSAUBD, GMEXICOB, ALFAA, BIMBOA")
        run_opt = st.form_submit_button("RUN OPTIMIZATION")
        
    if run_opt:
        with st.spinner("CALCULATING MATRICES..."):
            prices, _ = fetch_market_data(port_tickers, "3y")
            if prices is not None:
                assets = prices.drop(columns=["^MXX"])
                ret = np.log(assets / assets.shift(1)).dropna()
                
                col_opt, col_heat = st.columns(2)
                
                with col_opt:
                    st.markdown("### MAX SHARPE ALLOCATION")
                    weights = optimize_portfolio(ret, rf_total)
                    fig_p = px.pie(values=weights, names=assets.columns, color_discrete_sequence=px.colors.sequential.Aggrnyl)
                    fig_p.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00')
                    st.plotly_chart(fig_p, use_container_width=True)
                    
                with col_heat:
                    st.markdown("### CORRELATION MATRIX")
                    corr = ret.corr()
                    fig_h = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
                    fig_h.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00FF00')
                    st.plotly_chart(fig_h, use_container_width=True)

                st.markdown("### SWITCHING DIAGNOSIS (ALPHA BASED)")
                ret_market = np.log(prices["^MXX"] / prices["^MXX"].shift(1)).dropna()
                
                diag_data = []
                for t in assets.columns:
                    b, a = calc_beta_alpha(ret[t], ret_market, rf_daily)
                    action = "MANTENER" if a > 0 else "SWITCH"
                    diag_data.append({"Activo": t, "Beta": round(b, 2), "Alfa Anual": f"{a:.2%}", "Recomendacion": action})
                
                st.dataframe(pd.DataFrame(diag_data), use_container_width=True)

elif menu == "[HELP/EDU]":
    st.markdown("## [HELP/EDU] MARCO TEORICO")
    st.markdown("---")
    st.markdown("""
    * **Filosofia de Riesgo:** Base matematica en evaluacion de proyectos (Sapag Chain) y riesgo de simulacion (Velez Pareja).
    * **Beta:** Sensibilidad del activo frente al S&P/BMV IPC.
    * **Alfa de Jensen:** Exceso de retorno ajustado por riesgo sistémico.
    * **Altman Z-Score:** Probabilidad estadistica de quiebra.
    * **VaR Monte Carlo:** Maxima perdida esperada con un 95% de confianza.
    """)
