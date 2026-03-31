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
COLOR_ACENTO = "#00A4A6"  # Turquesa Institucional
COLOR_TEXTO = "#E0E0E0"
COLOR_VERDE = "#28A745"
COLOR_ROJO = "#DC3545"

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
    
    /* Tablas */
    .stDataFrame {{ background-color: {COLOR_PANEL}; border-radius: 8px; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SISTEMA DE MEMORIA (STATE MANAGEMENT)
# ==========================================
if 'cartera' not in st.session_state:
    st.session_state['cartera'] = []  # Lista de diccionarios para guardar activos

# ==========================================
# 3. MOTORES ANALITICOS Y DE DATOS
# ==========================================
@st.cache_data(ttl=300) # Se actualiza cada 5 minutos
def fetch_market_data(tickers_list, period="5y"):
    if not tickers_list: return None
    
    # Asegurar que siempre bajamos el IPC para calcular Betas
    if "^MXX" not in tickers_list:
        tickers_list.append("^MXX")
        
    data = yf.download(tickers_list, period=period, progress=False)
    if data.empty: return None
        
    prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers_list[0])
        
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

menu = st.sidebar.radio("Navegacion Principal:", [
    "Analisis Individual de Acciones", 
    "Analisis de Portafolio Completo"
])

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
    st.markdown("Busque un activo especifico para evaluar su rendimiento historico y metricas de riesgo antes de agregarlo a su portafolio.")
    
    with st.form("eq_form"):
        col_t, col_b = st.columns([3, 1])
        with col_t:
            ticker_ind = st.text_input("Ingrese el Ticker de la BMV (ej. WALMEX, ALFAA):", "WALMEX")
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
                
                # Grafico Tecnico
                df_ta = yf.download(t_str, period="1y", progress=False)
                if isinstance(df_ta, pd.Series): df_ta = df_ta.to_frame()
                df_ta = calc_bbands_pure(df_ta)
                
                st.markdown("### Comportamiento Tecnico (1 Ano)")
                fig = go.Figure(data=[go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'], low=df_ta['Low'], close=df_ta['Close'], name="Precio")])
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BBU'], line=dict(color='rgba(0,164,166,0.6)', dash='dash'), name="Banda Superior"))
                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BBL'], line=dict(color='rgba(0,164,166,0.6)', dash='dash'), name="Banda Inferior"))
                fig.update_layout(plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO, xaxis_rangeslider_visible=False, margin=dict(t=30, b=0, l=0, r=0))
                fig.update_xaxes(showgrid=True, gridcolor='#2A2D35')
                fig.update_yaxes(showgrid=True, gridcolor='#2A2D35')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se encontraron datos. Verifique el nombre del Ticker.")

# ==========================================
# 6. MODULO 2: GESTION DE PORTAFOLIO COMPLETO
# ==========================================
elif menu == "Analisis de Portafolio Completo":
    st.markdown(f"<h2 style='color: {COLOR_ACENTO};'>Consolidado de Portafolio</h2>", unsafe_allow_html=True)
    
    # 6.1 Constructor de Portafolio (Inputs)
    with st.expander("➕ AGREGAR ACTIVOS AL PORTAFOLIO", expanded=True):
        with st.form("add_asset_form"):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                nuevo_ticker = st.text_input("Ticker (ej. BIMBOA)")
            with c2:
                nuevas_acciones = st.number_input("Cant. Acciones", min_value=1.0, step=1.0)
            with c3:
                nuevo_precio = st.number_input("Precio de Compra ($)", min_value=0.01, step=1.0)
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                btn_agregar = st.form_submit_button("Agregar Activo")
                
            if btn_agregar and nuevo_ticker:
                t_str = nuevo_ticker.strip().upper() + ".MX" if not nuevo_ticker.strip().upper().endswith(".MX") else nuevo_ticker.strip().upper()
                st.session_state['cartera'].append({
                    "Ticker": t_str,
                    "Acciones": nuevas_acciones,
                    "Precio_Compra": nuevo_precio
                })
                st.rerun()

    # Boton para limpiar cartera
    if len(st.session_state['cartera']) > 0:
        if st.button("Limpiar Portafolio", type="secondary"):
            st.session_state['cartera'] = []
            st.rerun()

    # 6.2 Procesamiento y Tabla de Resultados
    if len(st.session_state['cartera']) == 0:
        st.info("Su portafolio esta vacio. Agregue activos utilizando el panel superior.")
    else:
        # Extraer lista unica de tickers
        tickers_cartera = list(set([item["Ticker"] for item in st.session_state['cartera']]))
        
        with st.spinner("Conectando al mercado para valuar su portafolio en tiempo real..."):
            precios_historicos = fetch_market_data(tickers_cartera, "3y")
            
            if precios_historicos is not None:
                # Consolidar datos de la cartera
                df_cartera = pd.DataFrame(st.session_state['cartera'])
                # Agrupar por si el usuario metio la misma accion dos veces
                df_cartera = df_cartera.groupby("Ticker").agg({"Acciones": "sum", "Precio_Compra": "mean"}).reset_index()
                
                # Calcular metricas financieras por activo
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
                    
                    # Saldo Inicial
                    s_inicial = acciones * p_compra
                    saldo_inicial_total += s_inicial
                    
                    # Saldo Actual (Mercado)
                    p_actual = precios_historicos[t].iloc[-1] if t in precios_historicos.columns else p_compra
                    s_actual = acciones * p_actual
                    saldo_actual_total += s_actual
                    
                    # Rendimiento Latente
                    rend_dinero = s_actual - s_inicial
                    rend_pct = (s_actual / s_inicial) - 1
                    
                    # Matematicas de Riesgo
                    beta, alpha = calc_beta_alpha_pure(retornos[t], ret_mercado, rf_daily) if t in retornos.columns else (0, 0)
                    capm = rf_total + beta * (rend_mercado_anual - rf_total)
                    
                    betas_dict[t] = beta
                    capm_dict[t] = capm
                    
                    filas_tabla.append({
                        "Activo": t,
                        "Acciones": acciones,
                        "Precio Compra": f"${p_compra:,.2f}",
                        "Precio Actual": f"${p_actual:,.2f}",
                        "Saldo Inicial": s_inicial,
                        "Saldo Actual": s_actual,
                        "Rendimiento": rend_pct,
                        "Beta": beta
                    })
                
                # Crear DataFrame final
                df_resumen = pd.DataFrame(filas_tabla)
                
                # Calcular Pesos (%) del Portafolio basados en Saldo Actual
                df_resumen["Peso %"] = df_resumen["Saldo Actual"] / saldo_actual_total
                
                # Calcular Metricas Ponderadas del Portafolio Completo
                beta_portafolio = sum(df_resumen["Peso %"] * df_resumen["Beta"])
                capm_portafolio = sum(df_resumen["Peso %"] * df_resumen["Activo"].map(capm_dict))
                rendimiento_global_pct = (saldo_actual_total / saldo_inicial_total) - 1
                
                # --- DASHBOARD VISUAL ---
                st.markdown("### Resumen de Saldos")
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Saldo Inicial (Invertido)", f"${saldo_inicial_total:,.2f}")
                col_s2.metric("Saldo Actual (Valuacion)", f"${saldo_actual_total:,.2f}", f"{rendimiento_global_pct:.2%} Global")
                col_s3.metric("Ganancia / Perdida", f"${(saldo_actual_total - saldo_inicial_total):,.2f}")
                
                st.markdown("### Metricas de Riesgo del Portafolio (Ponderadas)")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Beta Ponderada", f"{beta_portafolio:.2f}")
                col_r2.metric("CAPM Esperado", f"{capm_portafolio:.2%}")
                col_r3.metric("Tasa de Referencia (Rf)", f"{rf_total:.2%}")
                
                st.markdown("---")
                col_grafica, col_tabla = st.columns([1, 1.5])
                
                with col_grafica:
                    st.markdown("#### Composicion del Portafolio")
                    fig_pie = px.pie(df_resumen, values='Saldo Actual', names='Activo', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
                    fig_pie.update_layout(plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_tabla:
                    st.markdown("#### Desglose de Activos")
                    # Formatear columnas monetarias para visualizacion
                    df_visual = df_resumen.copy()
                    df_visual["Saldo Inicial"] = df_visual["Saldo Inicial"].apply(lambda x: f"${x:,.2f}")
                    df_visual["Saldo Actual"] = df_visual["Saldo Actual"].apply(lambda x: f"${x:,.2f}")
                    df_visual["Rendimiento"] = df_visual["Rendimiento"].apply(lambda x: f"{x:.2%}")
                    df_visual["Peso %"] = df_visual["Peso %"].apply(lambda x: f"{x:.2%}")
                    df_visual["Beta"] = df_visual["Beta"].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(df_visual[["Activo", "Acciones", "Precio Compra", "Precio Actual", "Saldo Actual", "Rendimiento", "Peso %"]], use_container_width=True)
            
            else:
                st.error("Ocurrio un error al descargar los precios. Verifique los tickers.")
