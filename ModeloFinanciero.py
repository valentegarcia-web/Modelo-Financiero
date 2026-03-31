import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. CONFIGURACION Y UX (MODERN DARK CORPORATIVO)
# ==========================================
st.set_page_config(page_title="CONFIDELIS - Wealth Analytics", layout="wide", initial_sidebar_state="expanded")

COLOR_FONDO = "#0E1117"
COLOR_PANEL = "#1A1C23"
COLOR_ACENTO = "#00A4A6"  
COLOR_TEXTO = "#E0E0E0"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLOR_FONDO}; color: {COLOR_TEXTO}; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    h1, h2, h3, h4, h5, h6, p, span, label {{ color: {COLOR_TEXTO} !important; }}
    .stSidebar {{ background-color: {COLOR_PANEL} !important; border-right: 1px solid #2A2D35; }}
    hr {{ border-color: #2A2D35; }}
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{ background-color: {COLOR_FONDO} !important; color: #FFF !important; border: 1px solid #2A2D35 !important; border-radius: 4px; }}
    
    .stButton>button, [data-testid="baseButton-secondaryFormSubmit"] {{ 
        background-color: {COLOR_ACENTO} !important; 
        color: #FFFFFF !important; 
        border: none !important; 
        border-radius: 6px !important; 
        font-weight: 600; 
        width: 100%; 
        transition: 0.3s;
    }}
    .stButton>button:hover, [data-testid="baseButton-secondaryFormSubmit"]:hover {{ 
        background-color: #008C8D !important; 
    }}
    
    div[data-testid="metric-container"] {{ background-color: {COLOR_PANEL}; border: 1px solid #2A2D35; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_ACENTO} !important; font-size: 1.8rem !important; font-weight: bold; }}
    div[data-testid="stMetricLabel"] {{ color: #A0A0A0 !important; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 1px; }}
    
    .stDataFrame {{ background-color: {COLOR_PANEL}; border-radius: 8px; }}
    
    /* Estilo para el panel de Insights */
    .insight-box {{
        background-color: rgba(0, 164, 166, 0.1);
        border-left: 5px solid {COLOR_ACENTO};
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SISTEMA DE MEMORIA
# ==========================================
if 'cartera' not in st.session_state:
    st.session_state['cartera'] = []  

# ==========================================
# 3. MOTORES ANALITICOS
# ==========================================
@st.cache_data(ttl=300) 
def fetch_market_data(tickers_list, period="5y"):
    if not tickers_list: return None
    if "^MXX" not in tickers_list: tickers_list.append("^MXX")
        
    data = yf.download(tickers_list, period=period, progress=False)
    if data.empty: return None
        
    prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    if isinstance(prices, pd.Series): prices = prices.to_frame(name=tickers_list[0])
    prices = prices.ffill().bfill()
    return prices

def calc_beta_alpha_pure(stock_returns, market_returns, rf_daily):
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if df.empty or len(df) < 2: return 0, 0
    y = df.iloc[:, 0] - rf_daily
    x = df.iloc[:, 1] - rf_daily
    cov_matrix = np.cov(y, x)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha_daily = y.mean() - beta * x.mean()
    return beta, alpha_daily * 252

def calc_bbands_pure(df, window=20, num_std=2):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['BBU'] = df['SMA'] + (df['STD'] * num_std)
    df['BBL'] = df['SMA'] - (df['STD'] * num_std)
    return df

# ==========================================
# 4. ENRUTAMIENTO UI (SIDEBAR)
# ==========================================
st.sidebar.markdown(f"<h3 style='color: {COLOR_ACENTO};'>CONFIDELIS ANALYTICS</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navegacion Principal:", ["Analisis Individual de Acciones", "Analisis de Portafolio Completo"])
st.sidebar.markdown("---")
st.sidebar.markdown("#### Parametros Macro (Mexico)")
rf_input = st.sidebar.number_input("Tasa Libre de Riesgo (CETES) %", value=11.00, step=0.1) / 100
embi_input = st.sidebar.number_input("Riesgo Pais (EMBI) %", value=3.50, step=0.1) / 100
rf_total = rf_input + embi_input 
rf_daily = rf_total / 252

# ==========================================
# 5. MODULO 1: ANALISIS INDIVIDUAL
# ==========================================
if menu == "Analisis Individual de Acciones":
    st.markdown(f"<h2 style='color: {COLOR_ACENTO};'>Analisis Individual (Equity)</h2>", unsafe_allow_html=True)
    with st.form("eq_form"):
        col_t, col_b = st.columns([3, 1])
        with col_t: ticker_ind = st.text_input("Ingrese el Ticker de la BMV (ej. WALMEX, ALFAA):", "WALMEX")
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            run_des = st.form_submit_button("Analizar Activo")
            
    if run_des:
        t_str = ticker_ind.strip().upper() + ".MX" if not ticker_ind.strip().upper().endswith(".MX") else ticker_ind.strip().upper()
        with st.spinner(f"Descargando datos del mercado para {t_str}..."):
            prices = fetch_market_data([t_str], "3y")
            if prices is not None and t_str in prices.columns:
                ret = np.log(prices / prices.shift(1)).dropna()
                beta, alpha = calc_beta_alpha_pure(ret[t_str], ret["^MXX"], rf_daily)
                capm_e = rf_total + beta * ((ret["^MXX"].mean()*252) - rf_total)
                precio_actual = prices[t_str].iloc[-1]
                
                st.markdown("### Metricas Clave")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precio Actual", f"${precio_actual:.2f}")
                c2.metric("Beta (Riesgo)", f"{beta:.2f}")
                c3.metric("Alfa (Exceso Retorno)", f"{alpha:.2%}")
                c4.metric("CAPM (Retorno Esperado)", f"{capm_e:.2%}")
                
                df_ta = yf.download(t_str, period="1y", progress=False)
                if isinstance(df_ta, pd.Series): df_ta = df_ta.to_frame()
                df_ta = calc_bbands_pure(df_ta)
                
                fig = go.Figure(data=[go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'], low=df_ta['Low'], close=df_ta['Close'], name="Precio")])
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BBU'], line=dict(color='rgba(0,164,166,0.6)', dash='dash'), name="Banda Superior"))
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BBL'], line=dict(color='rgba(0,164,166,0.6)', dash='dash'), name="Banda Inferior"))
                fig.update_layout(plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO, xaxis_rangeslider_visible=False, margin=dict(t=30, b=0, l=0, r=0))
                fig.update_xaxes(showgrid=True, gridcolor='#2A2D35')
                fig.update_yaxes(showgrid=True, gridcolor='#2A2D35')
                st.plotly_chart(fig, use_container_width=True)
            else: st.error("No se encontraron datos.")

# ==========================================
# 6. MODULO 2: GESTION DE PORTAFOLIO COMPLETO
# ==========================================
elif menu == "Analisis de Portafolio Completo":
    st.markdown(f"<h2 style='color: {COLOR_ACENTO};'>Consolidado de Portafolio y Riesgos</h2>", unsafe_allow_html=True)
    
    with st.expander("➕ AGREGAR ACTIVOS AL PORTAFOLIO", expanded=True):
        with st.form("add_asset_form"):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1: nuevo_ticker = st.text_input("Ticker (ej. BIMBOA)")
            with c2: nuevas_acciones = st.number_input("Cant. Acciones", min_value=1.0, step=1.0)
            with c3: nuevo_precio = st.number_input("Precio de Compra ($)", min_value=0.01, step=1.0)
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                btn_agregar = st.form_submit_button("Agregar Activo")
                
            if btn_agregar and nuevo_ticker:
                t_str = nuevo_ticker.strip().upper() + ".MX" if not nuevo_ticker.strip().upper().endswith(".MX") else nuevo_ticker.strip().upper()
                st.session_state['cartera'].append({"Ticker": t_str, "Acciones": nuevas_acciones, "Precio_Compra": nuevo_precio})
                st.rerun()

    if len(st.session_state['cartera']) > 0:
        if st.button("Limpiar Portafolio"):
            st.session_state['cartera'] = []
            st.rerun()

    if len(st.session_state['cartera']) == 0:
        st.info("Su portafolio esta vacio. Agregue activos utilizando el panel superior.")
    else:
        tickers_cartera = list(set([item["Ticker"] for item in st.session_state['cartera']]))
        
        with st.spinner("Conectando al mercado para valuar su portafolio en tiempo real..."):
            precios_historicos = fetch_market_data(tickers_cartera, "3y")
            
            if precios_historicos is not None:
                df_cartera = pd.DataFrame(st.session_state['cartera'])
                df_cartera = df_cartera.groupby("Ticker").agg({"Acciones": "sum", "Precio_Compra": "mean"}).reset_index()
                
                saldo_inicial_total = 0
                saldo_actual_total = 0
                filas_tabla = []
                betas_dict = {}
                capm_dict = {}
                
                retornos = np.log(precios_historicos / precios_historicos.shift(1)).dropna()
                ret_mercado = retornos["^MXX"]
                rend_mercado_anual = ret_mercado.mean() * 252
                
                for index, row in df_cartera.iterrows():
                    t = row["Ticker"]
                    acciones = row["Acciones"]
                    p_compra = row["Precio_Compra"]
                    
                    s_inicial = acciones * p_compra
                    saldo_inicial_total += s_inicial
                    
                    p_actual = precios_historicos[t].iloc[-1] if t in precios_historicos.columns else p_compra
                    s_actual = acciones * p_actual
                    saldo_actual_total += s_actual
                    
                    rend_pct = (s_actual / s_inicial) - 1
                    beta, alpha = calc_beta_alpha_pure(retornos[t], ret_mercado, rf_daily) if t in retornos.columns else (0, 0)
                    capm = rf_total + beta * (rend_mercado_anual - rf_total)
                    
                    betas_dict[t] = beta
                    capm_dict[t] = capm
                    
                    filas_tabla.append({
                        "Activo": t, "Acciones": acciones, "Precio Compra": p_compra, "Precio Actual": p_actual,
                        "Saldo Inicial": s_inicial, "Saldo Actual": s_actual, "Rendimiento": rend_pct,
                        "Beta": beta, "Alfa": alpha
                    })
                
                df_resumen = pd.DataFrame(filas_tabla)
                df_resumen["Peso %"] = df_resumen["Saldo Actual"] / saldo_actual_total
                
                # --- MATEMATICAS AVANZADAS ---
                beta_portafolio = sum(df_resumen["Peso %"] * df_resumen["Beta"])
                capm_portafolio = sum(df_resumen["Peso %"] * df_resumen["Activo"].map(capm_dict))
                rendimiento_global_pct = (saldo_actual_total / saldo_inicial_total) - 1
                
                activos_validos = [t for t in df_resumen["Activo"].tolist() if t in retornos.columns]
                pesos_validos = df_resumen.set_index("Activo").loc[activos_validos, "Peso %"].values
                
                if len(activos_validos) > 0:
                    ret_cartera_diario = (retornos[activos_validos] * pesos_validos).sum(axis=1)
                    port_vol_annual = ret_cartera_diario.std() * np.sqrt(252)
                    port_ret_annual = ret_cartera_diario.mean() * 252
                    var_95_mxn = saldo_actual_total * (1.645 * port_vol_annual)
                    sharpe = (port_ret_annual - rf_total) / port_vol_annual if port_vol_annual > 0 else 0
                else:
                    port_vol_annual = sharpe = var_95_mxn = 0

                # -----------------------------------------------------------
                # EL CEREBRO ANALITICO: RESUMEN INTELIGENTE
                # -----------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🤖 Diagnostico Automático del Portafolio")
                
                # Encontrar estrellas y rezagados
                mejor_activo = df_resumen.loc[df_resumen['Rendimiento'].idxmax()]
                peor_activo = df_resumen.loc[df_resumen['Rendimiento'].idxmin()]
                mas_riesgoso = df_resumen.loc[df_resumen['Beta'].idxmax()]
                activos_toxicos = df_resumen[(df_resumen['Alfa'] < 0) & (df_resumen['Beta'] > 1)]
                
                mensaje_diagnostico = f"**Estado General:** Su portafolio tiene un rendimiento global de **{rendimiento_global_pct:.2%}** y una volatilidad anual estimada del **{port_vol_annual:.2%}**.<br><br>"
                mensaje_diagnostico += f"🚀 **El Motor del Portafolio:** **{mejor_activo['Activo']}** es su mejor activo con una ganancia de **{mejor_activo['Rendimiento']:.2%}**.<br>"
                mensaje_diagnostico += f"⚠️ **Alerta de Riesgo:** **{mas_riesgoso['Activo']}** es el activo más agresivo de su cartera (Beta: {mas_riesgoso['Beta']:.2f}). Sus fluctuaciones impactarán fuertemente su saldo.<br>"
                
                if not activos_toxicos.empty:
                    toxicos_str = ", ".join(activos_toxicos['Activo'].tolist())
                    mensaje_diagnostico += f"<br>🔴 **RECOMENDACION DE SWITCH:** Se detectó que **{toxicos_str}** tienen un *Alfa negativo* a pesar de su alto riesgo. Le sugerimos evaluar la venta de estas posiciones y rotar el capital hacia activos con mejor perfil."
                else:
                    mensaje_diagnostico += "<br>🟢 **Salud Optimista:** Ningún activo de alto riesgo está destruyendo valor (Alfa positivo en activos volátiles). Mantenga su estrategia actual."

                st.markdown(f"<div class='insight-box'>{mensaje_diagnostico}</div>", unsafe_allow_html=True)
                
                # --- DASHBOARD VISUAL ---
                st.markdown("### Resumen de Cuenta")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Capital Invertido", f"${saldo_inicial_total:,.2f}")
                col_s2.metric("Valuacion de Mercado", f"${saldo_actual_total:,.2f}", f"{rendimiento_global_pct:.2%} Global")
                col_s3.metric("Ganancia / Perdida", f"${(saldo_actual_total - saldo_inicial_total):,.2f}")
                col_s4.metric("VaR 95% (Riesgo de Caida)", f"-${var_95_mxn:,.2f}")

                # --- SECCION DE GRAFICOS AVANZADOS ---
                st.markdown("---")
                st.markdown("### Análisis Gráfico")
                
                tab1, tab2, tab3 = st.tabs(["📉 Backtest Histórico", "🎯 Riesgo vs Retorno", "🍩 Composición Ponderada"])
                
                with tab1:
                    # Grafica de retornos acumulados vs IPC
                    if len(activos_validos) > 0:
                        cum_ret_port = (1 + ret_cartera_diario).cumprod()
                        cum_ret_bench = (1 + ret_mercado).cumprod()
                        
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(x=cum_ret_port.index, y=cum_ret_port, name="Su Portafolio", line=dict(color=COLOR_ACENTO, width=3)))
                        fig_line.add_trace(go.Scatter(x=cum_ret_bench.index, y=cum_ret_bench, name="S&P/BMV IPC", line=dict(color="#888888", width=2, dash='dot')))
                        fig_line.update_layout(title="Crecimiento de $1 invertido (Portafolio vs Mercado)", plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO, yaxis_title="Multiplicador de Capital", xaxis_title="Fecha")
                        fig_line.update_xaxes(showgrid=True, gridcolor='#2A2D35')
                        fig_line.update_yaxes(showgrid=True, gridcolor='#2A2D35')
                        st.plotly_chart(fig_line, use_container_width=True)
                
                with tab2:
                    # Scatter Plot de Riesgo (Beta) vs Retorno (Rendimiento)
                    fig_scatter = px.scatter(
                        df_resumen, x="Beta", y="Rendimiento", size="Saldo Actual", color="Activo", 
                        hover_name="Activo", text="Activo", title="Mapa de Riesgo y Eficiencia"
                    )
                    fig_scatter.update_traces(textposition='top center', marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                    fig_scatter.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="red", annotation_text="Mercado Neutral")
                    fig_scatter.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="gray")
                    fig_scatter.update_layout(plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO, xaxis_title="Riesgo (Beta)", yaxis_title="Rendimiento Real (%)")
                    fig_scatter.update_xaxes(showgrid=True, gridcolor='#2A2D35')
                    fig_scatter.update_yaxes(showgrid=True, gridcolor='#2A2D35', tickformat='.1%')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption("Interpretación: Busque tener sus burbujas más grandes (mayor capital) en la zona superior izquierda (Alto rendimiento, Bajo riesgo). Burbujas en la zona inferior derecha son activos tóxicos.")

                with tab3:
                    # Grafica de Pastel
                    fig_pie = px.pie(df_resumen, values='Saldo Actual', names='Activo', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
                    fig_pie.update_layout(plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # --- TABLA DE DATOS ---
                st.markdown("---")
                st.markdown("#### Desglose de Operaciones")
                df_visual = df_resumen.copy()
                df_visual["Precio Compra"] = df_visual["Precio Compra"].apply(lambda x: f"${x:,.2f}")
                df_visual["Precio Actual"] = df_visual["Precio Actual"].apply(lambda x: f"${x:,.2f}")
                df_visual["Saldo Actual"] = df_visual["Saldo Actual"].apply(lambda x: f"${x:,.2f}")
                df_visual["Rendimiento"] = df_visual["Rendimiento"].apply(lambda x: f"{x:.2%}")
                df_visual["Peso %"] = df_visual["Peso %"].apply(lambda x: f"{x:.2%}")
                df_visual["Beta"] = df_visual["Beta"].apply(lambda x: f"{x:.2f}")
                df_visual["Alfa"] = df_visual["Alfa"].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(df_visual[["Activo", "Acciones", "Precio Compra", "Precio Actual", "Saldo Actual", "Rendimiento", "Peso %", "Beta", "Alfa"]], use_container_width=True)
            
            else:
                st.error("Ocurrio un error al descargar los precios. Verifique los tickers.")
