import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t, skew, kurtosis, norm
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge

# ==========================================
# WARNINGS — MANEJO SELECTIVO
# ==========================================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ==========================================
# 1. CONFIGURACION Y UX
# ==========================================
st.set_page_config(page_title="CONFIDELIS - Production Engine", layout="wide", initial_sidebar_state="expanded")

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
    .stButton>button {{ background-color: {COLOR_ACENTO} !important; color: #FFFFFF !important; border: none !important; border-radius: 6px !important; font-weight: 600; width: 100%; transition: 0.3s; }}
    .stButton>button:hover {{ background-color: #008C8D !important; }}
    div[data-testid="metric-container"] {{ background-color: {COLOR_PANEL}; border: 1px solid #2A2D35; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_ACENTO} !important; font-size: 1.5rem !important; font-weight: bold; }}
    .insight-box {{ background-color: rgba(0, 164, 166, 0.1); border-left: 5px solid {COLOR_ACENTO}; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
    .alert-box {{ background-color: rgba(220, 53, 69, 0.1); border-left: 5px solid #DC3545; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
    .badge-success {{ background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
    .badge-warning {{ background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

if 'cartera' not in st.session_state: st.session_state['cartera'] = []  
if 'meta_model' not in st.session_state: st.session_state['meta_model'] = None

# ==========================================
# 2. DATA ENGINE
# ==========================================
def fetch_market_data(tickers_list, period="5y", use_cache=True):
    """use_cache=True para backtesting, use_cache=False para ejecución live T+1"""
    if not tickers_list: return None, None
    tickers = tickers_list.copy()
    if "^MXX" not in tickers: tickers.append("^MXX")
    data = yf.download(tickers, period=period, progress=False)
    if data.empty: return None, None
    prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    volumes = data['Volume'] if 'Volume' in data else pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        volumes = volumes.to_frame(name=tickers[0])
    return prices.ffill().bfill(), volumes.ffill().bfill()

@st.cache_data(ttl=3600)
def fetch_market_data_cached(tickers_list, period="5y"):
    return fetch_market_data(tickers_list, period, use_cache=True)

def calcular_retornos_robustos(prices):
    returns = prices.pct_change().dropna()
    lower = returns.expanding(min_periods=30).quantile(0.01)
    lower = lower.combine_first(returns.rolling(252, min_periods=30).quantile(0.01))
    upper = returns.expanding(min_periods=30).quantile(0.99)
    upper = upper.combine_first(returns.rolling(252, min_periods=30).quantile(0.99))
    return returns.clip(lower=lower, upper=upper).fillna(0)

# ==========================================
# 3. FEATURE ENGINEERING & MULTI-FACTOR ALPHA
# ==========================================
def generate_multi_factor_alpha(prices_df):
    if prices_df.empty: return pd.Series(0, index=prices_df.columns)
    returns = prices_df.pct_change().dropna()
    volatilities = returns.std() * np.sqrt(252)
    
    mom_6m = (prices_df.iloc[-1] / prices_df.iloc[-126]) - 1 if len(prices_df) >= 126 else pd.Series(0, index=prices_df.columns)
    rank_mom = mom_6m.rank(pct=True) 
    vol_adj_mom = mom_6m / (volatilities + 1e-6)
    vol_60 = returns.rolling(60).std().iloc[-1] if len(returns) >= 60 else pd.Series(0.1, index=prices_df.columns)
    low_vol_factor = -vol_60.rank(pct=True)
    ma_50 = prices_df.rolling(50).mean().iloc[-1] if len(prices_df) >= 50 else prices_df.iloc[-1]
    ma_200 = prices_df.rolling(200).mean().iloc[-1] if len(prices_df) >= 200 else prices_df.iloc[-1]
    trend_factor = ((ma_50 / ma_200) - 1).rank(pct=True)
    
    signals = {}
    for col in prices_df.columns:
        if col == "^MXX": continue
        if len(returns[col]) >= 20:
            roll_mean = returns[col].rolling(20).mean().iloc[-1]
            roll_std = returns[col].rolling(20).std().iloc[-1]
            z_score = (returns[col].iloc[-1] - roll_mean) / roll_std if roll_std > 0 else 0
        else: z_score = 0
        signals[col] = (0.35 * rank_mom[col]) + (0.25 * vol_adj_mom[col]) + (0.15 * trend_factor[col]) + (0.15 * low_vol_factor[col]) + (0.10 * (-z_score / 3))
        
    signals_series = pd.Series(signals)
    if signals_series.std() > 0: return (signals_series - signals_series.mean()) / signals_series.std()
    return signals_series - signals_series.mean()

@st.cache_data(
    ttl=3600,
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: (df.shape, str(df.index[-1]) if len(df) > 0 else '')}
)
def generate_panel_ml_alpha(prices_df, returns_df, volumes_df):
    panel_data = []
    cross_sectional_mean = returns_df.mean(axis=1)
    latest_features_dict = {}
    
    for col in prices_df.columns:
        if col == "^MXX": continue
        df = pd.DataFrame(index=prices_df.index)
        df['mom_5'] = prices_df[col].pct_change(5)
        df['mom_20'] = prices_df[col].pct_change(20)
        df['mom_60'] = prices_df[col].pct_change(60)
        df['vol_20'] = returns_df[col].rolling(20).std()
        df['zscore_20'] = (returns_df[col] - returns_df[col].rolling(20).mean()) / (df['vol_20'] + 1e-6)
        df['vol_shock'] = volumes_df[col] / (volumes_df[col].rolling(20).mean() + 1e-6)
        df['dispersion'] = returns_df[col] - cross_sectional_mean
        df['autocorr_20'] = returns_df[col].rolling(20).apply(lambda x: x.autocorr() if len(x)>2 else 0, raw=False)
        
        df['target'] = returns_df[col].shift(-5).rolling(5).sum()
        df['asset'] = col
        
        latest_features_dict[col] = df.drop(columns=['target', 'asset']).iloc[-2]
        panel_data.append(df)
        
    df_all = pd.concat(panel_data)
    df_train = df_all.dropna(subset=['target']).iloc[:-5] # Embargo
    
    if len(df_train) < 100:
        return pd.Series(0, index=[col for col in prices_df.columns if col != "^MXX"])
        
    X_train = df_train.drop(columns=['target', 'asset'])
    y_train = df_train['target']
    
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.05, l2_regularization=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    X_today = pd.DataFrame(latest_features_dict).T
    preds = model.predict(X_today)
    
    ml_series = pd.Series(preds, index=X_today.index).rank(pct=True)
    if ml_series.std() > 0: return (ml_series - ml_series.mean()) / ml_series.std()
    return ml_series - ml_series.mean()

# ==========================================
# 4. RISK ENGINE & METRICS
# ==========================================
def get_ledoit_wolf_cov(returns):
    lw = LedoitWolf()
    lw.fit(returns.fillna(0))
    return lw.covariance_ * 252

def calculate_betas(returns_df, market_returns):
    X = sm.add_constant(market_returns.values)
    Y = returns_df.values
    betas = []
    for i in range(Y.shape[1]):
        model = sm.OLS(Y[:, i], X).fit()
        betas.append(model.params[1])
    return np.array(betas)

def detect_regime_gmm(market_returns):
    if len(market_returns) < 126: return "NEUTRAL"
    df_gmm = pd.DataFrame(index=market_returns.index)
    df_gmm['ret'] = market_returns.rolling(5).sum()
    df_gmm['vol'] = market_returns.rolling(20).std() * np.sqrt(252)
    df_gmm = df_gmm.dropna()
    if len(df_gmm) < 30: return "NEUTRAL"
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df_gmm.iloc[:-1])
    state = gmm.predict(df_gmm.iloc[-1:])
    means = gmm.means_
    safe_state = 0 if means[0, 1] < means[1, 1] else 1
    return "BULL / STABLE" if state[0] == safe_state else "CRISIS / VOLATILE"

def calcular_drawdown_avanzado(retornos):
    if len(retornos) == 0: return pd.Series([0]), 0
    acumulado = (1 + retornos).cumprod()
    max_acum = acumulado.cummax()
    drawdown = (acumulado - max_acum) / max_acum
    return drawdown, drawdown.min()

def probabilistic_sharpe_ratio(port_returns, rf_daily, benchmark_sr=0.0):
    if len(port_returns) < 30: return 0, 0
    excess_ret = port_returns - rf_daily
    if np.isclose(excess_ret.std(), 0): return 0, 0
    sr_daily = excess_ret.mean() / excess_ret.std()
    sr_ann = sr_daily * np.sqrt(252)
    sk = skew(port_returns)
    kt = kurtosis(port_returns)
    n = len(port_returns)
    sr_std = np.sqrt((1 - sk*sr_daily + ((kt-1)/4)*sr_daily**2) / (n-1))
    psr = norm.cdf((sr_daily - (benchmark_sr/np.sqrt(252))) / sr_std)
    dsr = sr_ann * (1 - (sk/6)*sr_daily + ((kt-3)/24)*(sr_daily**2))
    return dsr, psr

# ==========================================
# 5. ROBUST OPTIMIZATION ENGINE
# ==========================================
def optimizar_market_neutral_pro(expected_returns, cov_matrix, current_weights, betas, adv_weights_max, vol_forecast_array, regime):
    num_assets = len(expected_returns)
    lambda_reg = 0.05 / num_assets
    
    np.random.seed(42)
    mc_scenarios = np.random.multivariate_normal(np.zeros(num_assets), cov_matrix, 2000)
    lambda_cvar = 6.0 if regime == "CRISIS / VOLATILE" else 2.0
    
    def neg_sharpe(weights):
        p_ret = np.sum(expected_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        penalty_l2 = lambda_reg * np.sum(weights**2)
        
        delta = np.abs(weights - current_weights)
        spread_cost = 0.0005 * np.sum(delta)
        impact_cost = 0.005 * np.sum((delta**1.5) * vol_forecast_array / (adv_weights_max + 1e-6))
        
        sim_losses = -(mc_scenarios @ weights)
        cvar_true = np.mean(np.sort(sim_losses)[-int(0.05 * 2000):])
        penalty_tail = lambda_cvar * max(cvar_true, 0)
        
        if np.isclose(p_vol, 0): return 0
        return -(p_ret) / p_vol + penalty_l2 + spread_cost + impact_cost + penalty_tail

    bounds = tuple((-mw, mw) for mw in adv_weights_max)
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w)}, 
        {'type': 'eq', 'fun': lambda w: np.dot(w, betas)}, 
        {'type': 'ineq', 'fun': lambda w: 1.5 - np.sum(np.abs(w))} 
    )
    
    res = minimize(neg_sharpe, current_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success:
        st.warning(f"⚠️ Optimizador no convergió: {res.message}. Usando pesos actuales.")
        return current_weights
    return res.x

# ==========================================
# 6. WALK-FORWARD BACKTESTING (META-MODEL REAL)
# ==========================================
def backtest_walk_forward_meta_model(prices, volumes, activos, rf_daily, initial_capital=1e6):
    returns = calcular_retornos_robustos(prices)
    dates = returns.index
    portfolio_returns = []
    bench_returns = []
    idx_out = []
    
    current_weights = np.zeros(len(activos))
    rebalance_freq = 21
    
    X_meta = []
    y_meta = []
    prev_preds = None
    meta_model_final = None 
    
    for i in range(252, len(dates) - rebalance_freq, rebalance_freq): 
        start_idx = max(0, i - 504)
        train_prices = prices.iloc[start_idx:i]
        train_returns = returns.iloc[start_idx:i]
        train_volumes = volumes.iloc[start_idx:i]
        
        test_returns = returns.iloc[i+1 : i+1+rebalance_freq]
        if test_returns.empty or len(train_prices) < 126: break
        
        # --- 1. RIDGE META-MODEL ZERO-LEAKAGE ---
        if prev_preds is not None:
            realized_return = train_returns[activos].iloc[-rebalance_freq:].sum().mean()
            X_meta.append(prev_preds)
            y_meta.append(realized_return)
            
        c_stat = generate_multi_factor_alpha(train_prices[activos]).fillna(0)
        c_ml = generate_panel_ml_alpha(train_prices[activos], train_returns[activos], train_volumes[activos]).fillna(0)
        
        regime = detect_regime_gmm(train_returns["^MXX"])
        
        if len(X_meta) > 5:
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(X_meta, y_meta)
            meta_model_final = meta_model 
            curr_X = pd.DataFrame({'stat': c_stat.values, 'ml': c_ml.values})
            meta_alpha = pd.Series(meta_model.predict(curr_X), index=activos)
        else:
            w_ml = 0.2 if regime == "CRISIS / VOLATILE" else 0.7
            meta_alpha = ((1 - w_ml) * c_stat) + (w_ml * c_ml)

        prev_preds = [c_stat.mean(), c_ml.mean()]
        
        # --- 2. MATRICES Y OPTIMIZACION ---
        vol_forecast = train_returns[activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
        exp_ret = meta_alpha.values * vol_forecast.values
        cov = get_ledoit_wolf_cov(train_returns[activos])
        betas = calculate_betas(train_returns[activos], train_returns["^MXX"])
        
        adv_mxn = (train_prices[activos].tail(20) * train_volumes[activos].tail(20)).mean().values
        adv_weights_max = np.clip((0.10 * adv_mxn) / initial_capital, 0.01, 0.30)
        
        raw_weights = optimizar_market_neutral_pro(exp_ret, cov, current_weights, betas, adv_weights_max, vol_forecast.values, regime)
        
        current_vol = np.sqrt(np.dot(raw_weights.T, np.dot(cov, raw_weights)))
        max_dd_running = 0 if len(portfolio_returns) == 0 else calcular_drawdown_avanzado(pd.Series(portfolio_returns))[1]
        
        target_vol = 0.05 if (regime == "CRISIS / VOLATILE" or max_dd_running < -0.10) else 0.15
        new_weights = raw_weights * (target_vol / current_vol) if current_vol > 0 else raw_weights
            
        delta = np.abs(new_weights - current_weights)
        total_cost = (0.0005 * np.sum(delta)) + (0.005 * np.sum((delta**1.5) * vol_forecast.values / (adv_weights_max + 1e-6)))
        daily_exec_cost = total_cost / 5.0
        
        for j in range(len(test_returns)):
            day_ret = np.dot(new_weights, test_returns[activos].iloc[j].values)
            if j < 5: day_ret -= daily_exec_cost 
            portfolio_returns.append(day_ret)
            bench_returns.append(test_returns["^MXX"].iloc[j])
            idx_out.append(test_returns.index[j])
            
        current_weights = new_weights
        
    return pd.Series(portfolio_returns, index=idx_out), meta_model_final, pd.Series(bench_returns, index=idx_out)

def monte_carlo_shuffle_test(returns, n_sim=50):
    results = []
    for _ in range(n_sim):
        shuffled = returns.sample(frac=1, replace=False).reset_index(drop=True)
        equity = (1 + shuffled).cumprod()
        results.append(equity.values)
    return results

# ==========================================
# UI & DASHBOARD
# ==========================================
st.sidebar.markdown(f"<h3 style='color: {COLOR_ACENTO};'>CITADEL ARCHITECTURE</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = st.sidebar.radio("Módulos:", ["Live Execution Desk", "Backtesting Engine"])
rf_input = st.sidebar.number_input("Tasa Libre Riesgo (Rf) %", value=11.00, step=0.1) / 100
rf_daily = rf_input / 252

if menu == "Live Execution Desk":
    st.markdown(f"<h2 style='color: {COLOR_ACENTO};'>Live Execution Desk (Actionable Orders)</h2>", unsafe_allow_html=True)
    
    with st.expander("UNIVERSO DE ACTIVOS (Añadir mínimo 5)", expanded=True):
        with st.form("add_asset_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: nuevo_ticker = st.text_input("Ticker (ej. WALMEX)")
            with c2: weight_actual = st.number_input("Peso Actual %", value=0.0, step=1.0) / 100
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                btn_agregar = st.form_submit_button("Añadir Activo")
                
            if btn_agregar and nuevo_ticker:
                t_str = nuevo_ticker.strip().upper() + ".MX" if not nuevo_ticker.strip().upper().endswith(".MX") else nuevo_ticker.strip().upper()
                st.session_state['cartera'].append({"Ticker": t_str, "Peso": weight_actual})
                st.rerun()

    if len(st.session_state['cartera']) > 0 and st.button("Limpiar Universo"):
        st.session_state['cartera'] = []
        st.rerun()

    if st.session_state['meta_model'] is not None:
        st.markdown("<span class='badge-success'>✅ Meta-Model Ridge Cargado en Memoria</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-warning'>⚠️ Meta-Model No Disponible. Usando Fallback por Régimen. Corra el Backtest primero.</span>", unsafe_allow_html=True)
    
    if len(st.session_state['cartera']) > 4:
        df_cart = pd.DataFrame(st.session_state['cartera']).groupby("Ticker").sum().reset_index()
        activos_brutos = df_cart["Ticker"].tolist()
        current_weights_dict = dict(zip(df_cart["Ticker"], df_cart["Peso"]))
        capital_operativo = st.number_input("Capital Operativo de la Cuenta ($ MXN)", value=10000000, step=1000000)
        
        if st.button("⚡ Generar Órdenes para Mañana (T+1)"):
            with st.spinner("Descargando mercado, ejecutando Machine Learning & Risk Optimization..."):
                precios_historicos, volumen_historico = fetch_market_data(activos_brutos, "5y", use_cache=False)
                
                if precios_historicos is None:
                    st.error("Error al descargar datos de Yahoo Finance. Verifique su conexión o los tickers ingresados.")
                    st.stop()
                
                valid_tickers = [col for col in precios_historicos.columns if precios_historicos[col].count() >= 252 or col == "^MXX"]
                excluidos = set(activos_brutos) - set(valid_tickers)
                if excluidos:
                    st.warning(f"Excluidos por falta de historial (mínimo 252 días): {', '.join(excluidos)}")
                
                activos = [a for a in activos_brutos if a in valid_tickers]
                if len(activos) < 5:
                    st.error("No hay suficientes activos válidos (mínimo 5) para ejecutar el modelo de Machine Learning.")
                    st.stop()
                
                current_weights = np.array([current_weights_dict[a] for a in activos])
                returns = calcular_retornos_robustos(precios_historicos)
                
                if len(precios_historicos) > 504:
                    train_prices = precios_historicos.iloc[-504:]
                    train_returns = returns.iloc[-504:]
                    train_volumes = volumen_historico.iloc[-504:]
                else:
                    train_prices, train_returns, train_volumes = precios_historicos, returns, volumen_historico
                
                alpha_stat = generate_multi_factor_alpha(train_prices[activos]).fillna(0)
                alpha_ml = generate_panel_ml_alpha(train_prices[activos], train_returns[activos], train_volumes[activos]).fillna(0)
                
                regime = detect_regime_gmm(train_returns["^MXX"])
                
                if st.session_state['meta_model'] is not None:
                    curr_X = pd.DataFrame({'stat': alpha_stat.values, 'ml': alpha_ml.values})
                    meta_alpha = pd.Series(st.session_state['meta_model'].predict(curr_X), index=activos)
                else:
                    w_ml = 0.2 if regime == "CRISIS / VOLATILE" else 0.7
                    meta_alpha = ((1 - w_ml) * alpha_stat) + (w_ml * alpha_ml)
                
                vol_forecast = train_returns[activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
                exp_ret = meta_alpha.values * vol_forecast.values
                
                cov = get_ledoit_wolf_cov(train_returns[activos])
                betas = calculate_betas(train_returns[activos], train_returns["^MXX"])
                
                adv_mxn = (train_prices[activos].tail(20) * train_volumes[activos].tail(20)).mean().values
                adv_weights_max = np.clip((0.10 * adv_mxn) / capital_operativo, 0.01, 0.30)
                
                target_vol = 0.08 if regime == "CRISIS / VOLATILE" else 0.15
                
                raw_weights = optimizar_market_neutral_pro(exp_ret, cov, current_weights, betas, adv_weights_max, vol_forecast.values, regime)
                
                current_vol = np.sqrt(np.dot(raw_weights.T, np.dot(cov, raw_weights)))
                new_weights = raw_weights * (target_vol / current_vol) if current_vol > 0 else raw_weights
                
                df_orders = pd.DataFrame({"Activo": activos, "Peso Actual": current_weights, "Peso Objetivo": new_weights})
                df_orders["Delta (Trade)"] = df_orders["Peso Objetivo"] - df_orders["Peso Actual"]
                
                def classify_action(delta):
                    if delta > 0.01: return "🟢 COMPRAR"
                    elif delta < -0.01: return "🔴 VENDER / SHORT"
                    else: return "⚪ MANTENER"
                    
                df_orders["Acción Requerida"] = df_orders["Delta (Trade)"].apply(classify_action)
                df_orders["Capital a Mover ($)"] = df_orders["Delta (Trade)"] * capital_operativo
                
                st.markdown("### 🛡️ Métricas del Portafolio Objetivo")
                c1, c2, c3 = st.columns(3)
                c1.metric("Volatilidad Esperada Anualizada", f"{current_vol:.2%}")
                beta_neta = np.dot(new_weights, betas)
                c2.metric("Beta Neta Proyectada", f"{beta_neta:.4f}")
                c3.metric("Régimen Detectado (GMM)", regime)
                
                if regime == "CRISIS / VOLATILE":
                    st.markdown("<div class='alert-box'>🚨 <b>CRISIS DETECTADA:</b> El mercado presenta inestabilidad sistémica. El target de volatilidad ha sido reducido automáticamente para preservar el capital.</div>", unsafe_allow_html=True)
                
                for col in ["Peso Actual", "Peso Objetivo", "Delta (Trade)"]: 
                    df_orders[col] = df_orders[col].apply(lambda x: f"{x:.2%}")
                df_orders["Capital a Mover ($)"] = df_orders["Capital a Mover ($)"].apply(lambda x: f"${x:,.2f}")
                
                st.markdown(f"### 📋 Tabla de Órdenes (T+1)")
                st.dataframe(df_orders.set_index("Activo").sort_values(by="Acción Requerida"), use_container_width=True)
                
                csv = df_orders.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Exportar Libro de Órdenes (CSV)", data=csv, file_name='execution_orders_T1.csv', mime='text/csv')

    else:
        st.info("Agregue al menos 5 activos y asigne sus pesos para generar el libro de órdenes.")

elif menu == "Backtesting Engine":
    st.markdown(f"<h2 style='color: {COLOR_ACENTO};'>Purged Walk-Forward Backtest</h2>", unsafe_allow_html=True)
    
    if len(st.session_state['cartera']) > 4:
        activos_brutos = list(set([item["Ticker"] for item in st.session_state['cartera']]))
        if st.button("⚡ Ejecutar Motor Cuantitativo Completo (Heavy Compute)"):
            with st.spinner("Procesando Panel ML, entrenando Ridge Meta-Model Causal y simulando Costos Reales..."):
                precios_historicos, volumen_historico = fetch_market_data_cached(activos_brutos, "5y")
                
                if precios_historicos is None:
                    st.error("Fallo al obtener datos históricos.")
                    st.stop()
                    
                valid_tickers = [col for col in precios_historicos.columns if precios_historicos[col].count() >= 252 or col == "^MXX"]
                activos = [a for a in activos_brutos if a in valid_tickers]
                
                oos_ret, trained_meta_model, bench_ret = backtest_walk_forward_meta_model(precios_historicos, volumen_historico, activos, rf_daily, initial_capital=10000000)
                
                if trained_meta_model is not None:
                    st.session_state['meta_model'] = trained_meta_model
                
                if len(oos_ret) > 0:
                    total_ret = np.prod(1 + oos_ret) - 1
                    ann_ret = (1 + total_ret) ** (252/len(oos_ret)) - 1
                    vol = oos_ret.std() * np.sqrt(252)
                    dsr, psr = probabilistic_sharpe_ratio(oos_ret, rf_daily)
                    
                    sharpe = (ann_ret - (rf_daily * 252)) / vol if vol > 0 else 0
                    drawdown, max_dd = calcular_drawdown_avanzado(oos_ret)
                    
                    st.markdown("##### 🏆 Métricas de Fondo Institucional (OOS)")
                    cm1, cm2, cm3, cm4, cm5, cm6 = st.columns(6)
                    cm1.metric("CAGR (Alpha)", f"{ann_ret:.2%}")
                    cm2.metric("Volatilidad OOS", f"{vol:.2%}")
                    cm3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    cm4.metric("Max Drawdown", f"{max_dd:.2%}")
                    cm5.metric("PSR", f"{psr:.2%}")
                    cm6.metric("DSR", f"{dsr:.2f}")
                    
                    equity = (1 + oos_ret).cumprod()
                    equity_bench = (1 + bench_ret).cumprod()
                    
                    fig_oos = go.Figure()
                    
                    shuffle_paths = monte_carlo_shuffle_test(oos_ret, n_sim=50)
                    for path in shuffle_paths:
                        path_len = len(path)
                        fig_oos.add_trace(go.Scatter(
                            x=equity.index[:path_len], 
                            y=path, 
                            mode='lines', 
                            line=dict(color='gray', width=1), 
                            opacity=0.15, 
                            showlegend=False
                        ))
                        
                    fig_oos.add_trace(go.Scatter(x=equity.index, y=equity, name="Ridge Ensemble L/S Strategy", line=dict(color=COLOR_ACENTO, width=3)))
                    fig_oos.add_trace(go.Scatter(x=equity_bench.index, y=equity_bench, name="IPC Benchmark (^MXX)", line=dict(color="#888888", width=2, dash='dot')))
                    fig_oos.update_layout(title="Curva de Capital Real con MC Shuffle Overlay (Neto de Slippage, CVaR Penalty, Beta=0)", plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO)
                    st.plotly_chart(fig_oos, use_container_width=True)
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name="Drawdown", line=dict(color="#DC3545", width=2)))
                    fig_dd.update_layout(title="Profundidad de Drawdown Histórico", plot_bgcolor=COLOR_FONDO, paper_bgcolor=COLOR_FONDO, font_color=COLOR_TEXTO)
                    fig_dd.update_yaxes(tickformat='.1%')
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                else: st.warning("Datos insuficientes para procesar el modelo matemático.")
    else:
        st.info("Vaya a la pestaña 'Live Execution Desk' y agregue al menos 5 activos para habilitar el Backtest.")
