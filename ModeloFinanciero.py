import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, norm
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from datetime import datetime
import base64
import os

# ==========================================
# WARNINGS
# ==========================================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ==========================================
# 1. CONFIGURACION — IDENTIDAD CONFIDELIS
# ==========================================
st.set_page_config(
    page_title="CONFIDELIS - Gestión Patrimonial Inteligente",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta Confidelis (Manual de Marca)
# Primarios del imagotipo: teal profundo + grises
# Se evitan: naranja #FF6900 y azul fuerte #13294B
TEAL_DARK = "#2B6E6E"
TEAL_MID = "#3D8C8C"
TEAL_LIGHT = "#5AACAC"
TEAL_GLOW = "#7ECFCF"
GRAY_900 = "#111518"
GRAY_800 = "#1A1F25"
GRAY_700 = "#242A32"
GRAY_600 = "#2E3640"
GRAY_400 = "#6B7A8D"
GRAY_300 = "#8C9BAD"
GRAY_200 = "#B0BEC5"
GRAY_100 = "#D6DEE5"
WHITE = "#F0F2F5"
SUCCESS = "#2E8B57"    # verde sobrio
DANGER = "#B04050"     # rojo sobrio (no brillante)
WARN_SOFT = "#C9A030"  # dorado suave

# Logo path
LOGO_PATH = "/Users/confidelis/Documents/LOGOS/Confidelis/Confidelis, EL aliado que tu pratimonio Merece.png"

def get_logo_base64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background-color: {GRAY_900};
        color: {GRAY_100};
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {WHITE} !important; font-weight: 600; letter-spacing: -0.02em; }}
    p, span, label, li {{ color: {GRAY_200} !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {GRAY_800} !important;
        border-right: 1px solid {GRAY_700};
    }}
    section[data-testid="stSidebar"] .stRadio label span {{
        color: {GRAY_200} !important;
        font-size: 0.95rem;
    }}

    hr {{ border-color: {GRAY_700}; }}

    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {{
        background-color: {GRAY_800} !important;
        color: {WHITE} !important;
        border: 1px solid {GRAY_600} !important;
        border-radius: 8px;
    }}
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
        border-color: {TEAL_MID} !important;
        box-shadow: 0 0 0 2px rgba(61,140,140,0.2);
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {TEAL_DARK}, {TEAL_MID}) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
        letter-spacing: 0.02em;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {TEAL_MID}, {TEAL_LIGHT}) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(61,140,140,0.3);
    }}

    /* Metrics */
    div[data-testid="metric-container"] {{
        background-color: {GRAY_800};
        border: 1px solid {GRAY_700};
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEAL_LIGHT} !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {GRAY_300} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {GRAY_800} !important;
        border-radius: 10px;
        color: {GRAY_100} !important;
    }}

    /* DataFrames */
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

    /* Custom boxes */
    .info-card {{
        background: linear-gradient(135deg, {GRAY_800}, {GRAY_700});
        border: 1px solid {GRAY_600};
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 16px;
    }}
    .info-card h4 {{ margin: 0 0 8px 0; font-size: 1rem; }}
    .info-card p {{ margin: 0; font-size: 0.9rem; line-height: 1.5; }}

    .alert-crisis {{
        background: linear-gradient(135deg, rgba(176,64,80,0.12), rgba(176,64,80,0.05));
        border-left: 4px solid {DANGER};
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
    }}
    .alert-stable {{
        background: linear-gradient(135deg, rgba(46,139,87,0.12), rgba(46,139,87,0.05));
        border-left: 4px solid {SUCCESS};
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
    }}
    .alert-info {{
        background: linear-gradient(135deg, rgba(61,140,140,0.12), rgba(61,140,140,0.05));
        border-left: 4px solid {TEAL_MID};
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}
    .badge-ok {{ background-color: rgba(46,139,87,0.2); color: {SUCCESS}; border: 1px solid rgba(46,139,87,0.3); }}
    .badge-warn {{ background-color: rgba(201,160,48,0.15); color: {WARN_SOFT}; border: 1px solid rgba(201,160,48,0.25); }}

    /* Asset pills */
    .asset-pill {{
        display: inline-block;
        background: linear-gradient(135deg, {GRAY_700}, {GRAY_600});
        border: 1px solid {GRAY_600};
        color: {TEAL_GLOW} !important;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Tooltip style */
    .explainer {{
        color: {GRAY_400} !important;
        font-size: 0.82rem;
        font-style: italic;
        margin-top: 4px;
        line-height: 1.4;
    }}

    /* Logo area */
    .logo-container {{
        text-align: center;
        padding: 20px 10px 10px 10px;
    }}
    .logo-container img {{
        max-width: 180px;
        opacity: 0.95;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {GRAY_800};
        border-radius: 8px;
        color: {GRAY_300} !important;
        border: 1px solid {GRAY_700};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {TEAL_DARK} !important;
        color: white !important;
    }}

    /* Download button */
    .stDownloadButton>button {{
        background: {GRAY_700} !important;
        border: 1px solid {GRAY_600} !important;
        color: {TEAL_LIGHT} !important;
    }}
    .stDownloadButton>button:hover {{
        background: {GRAY_600} !important;
        border-color: {TEAL_MID} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Session state
if 'cartera' not in st.session_state:
    st.session_state['cartera'] = []
if 'meta_model' not in st.session_state:
    st.session_state['meta_model'] = None

# ==========================================
# 2. DATA ENGINE
# ==========================================
POPULAR_MX_TICKERS = {
    "WALMEX.MX": "Walmart de México",
    "AMXB.MX": "América Móvil",
    "FEMSAUBD.MX": "FEMSA",
    "GMEXICOB.MX": "Grupo México",
    "GFNORTEO.MX": "Banorte",
    "TABORAB.MX": "Tabora (antes Bimbo)",
    "BIMBOA.MX": "Grupo Bimbo",
    "CEMEXCPO.MX": "CEMEX",
    "ASURB.MX": "ASUR (Aeropuertos)",
    "GCARSOA1.MX": "Grupo Carso",
    "AC.MX": "Arca Continental",
    "KIMBERA.MX": "Kimberly-Clark MX",
    "ELEKTRA.MX": "Elektra",
    "ALSEA.MX": "Alsea",
    "LABB.MX": "Genomma Lab",
    "OMAB.MX": "OMA (Aeropuertos)",
    "GRUMAB.MX": "Gruma",
    "MEGACPO.MX": "Megacable",
    "PABORAB.MX": "Promotora Ambiental",
    "BOLSAA.MX": "Bolsa Mexicana de Valores",
}


def fetch_market_data(tickers_list, period="5y"):
    """Descarga precios y volúmenes de Yahoo Finance."""
    if not tickers_list:
        return None, None
    tickers = tickers_list.copy()
    if "^MXX" not in tickers:
        tickers.append("^MXX")
    try:
        data = yf.download(tickers, period=period, progress=False)
    except Exception as e:
        st.error(f"No se pudo conectar con el proveedor de datos: {e}")
        return None, None
    if data.empty:
        return None, None
    prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    volumes = data['Volume'] if 'Volume' in data else pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        volumes = volumes.to_frame(name=tickers[0])
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No se encontraron datos para: {', '.join(missing)}")
    return prices.ffill().bfill(), volumes.ffill().bfill()


@st.cache_data(ttl=3600)
def fetch_market_data_cached(tickers_list, period="5y"):
    return fetch_market_data(tickers_list, period)


def calcular_retornos_robustos(prices, lookback_for_quantile=252):
    """Winsoriza retornos usando SOLO datos pasados (sin look-ahead bias)."""
    returns = prices.pct_change().dropna()
    if len(returns) < lookback_for_quantile:
        lookback_for_quantile = max(30, len(returns))
    lower = returns.rolling(lookback_for_quantile, min_periods=30).quantile(0.01)
    upper = returns.rolling(lookback_for_quantile, min_periods=30).quantile(0.99)
    clipped = returns.clip(lower=lower, upper=upper)
    clipped = clipped.fillna(returns)
    return clipped.fillna(0)


def get_asset_summary(prices_df, ticker):
    """Genera resumen legible de un activo para el usuario."""
    if ticker not in prices_df.columns or len(prices_df[ticker].dropna()) < 20:
        return None
    p = prices_df[ticker].dropna()
    last_price = p.iloc[-1]
    ret_1m = (p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0
    ret_3m = (p.iloc[-1] / p.iloc[-63] - 1) if len(p) >= 63 else 0
    ret_1y = (p.iloc[-1] / p.iloc[-252] - 1) if len(p) >= 252 else 0
    vol = p.pct_change().dropna().std() * np.sqrt(252)
    max_price = p.max()
    dist_from_max = (last_price / max_price - 1)
    return {
        "Precio": f"${last_price:,.2f}",
        "Rend. 1 Mes": f"{ret_1m:+.1%}",
        "Rend. 3 Meses": f"{ret_3m:+.1%}",
        "Rend. 1 Año": f"{ret_1y:+.1%}",
        "Volatilidad Anual": f"{vol:.1%}",
        "Dist. desde Máximo": f"{dist_from_max:+.1%}",
    }


# ==========================================
# 3. FEATURE ENGINEERING & MULTI-FACTOR ALPHA
# ==========================================
def generate_multi_factor_alpha(prices_df):
    if prices_df.empty:
        return pd.Series(0, index=prices_df.columns)
    returns = prices_df.pct_change().dropna()
    vol_126 = returns.rolling(126, min_periods=60).std().iloc[-1] * np.sqrt(126)
    mom_6m = (prices_df.iloc[-1] / prices_df.iloc[-126]) - 1 if len(prices_df) >= 126 else pd.Series(0, index=prices_df.columns)
    rank_mom = mom_6m.rank(pct=True)
    vol_adj_mom = mom_6m / (vol_126 + 1e-6)
    vol_60 = returns.rolling(60).std().iloc[-1] if len(returns) >= 60 else pd.Series(0.1, index=prices_df.columns)
    low_vol_factor = -vol_60.rank(pct=True)
    ma_50 = prices_df.rolling(50).mean().iloc[-1] if len(prices_df) >= 50 else prices_df.iloc[-1]
    ma_200 = prices_df.rolling(200).mean().iloc[-1] if len(prices_df) >= 200 else prices_df.iloc[-1]
    trend_factor = ((ma_50 / ma_200) - 1).rank(pct=True)

    signals = {}
    for col in prices_df.columns:
        if col == "^MXX":
            continue
        if len(returns[col]) >= 20:
            roll_mean = returns[col].rolling(20).mean().iloc[-1]
            roll_std = returns[col].rolling(20).std().iloc[-1]
            z_score = (returns[col].iloc[-1] - roll_mean) / roll_std if roll_std > 0 else 0
        else:
            z_score = 0
        signals[col] = (
            0.35 * rank_mom.get(col, 0)
            + 0.25 * vol_adj_mom.get(col, 0)
            + 0.15 * trend_factor.get(col, 0)
            + 0.15 * low_vol_factor.get(col, 0)
            + 0.10 * (-z_score / 3)
        )
    signals_series = pd.Series(signals)
    if signals_series.std() > 0:
        return (signals_series - signals_series.mean()) / signals_series.std()
    return signals_series - signals_series.mean()


def generate_panel_ml_alpha(prices_df, returns_df, volumes_df):
    panel_data = []
    cross_sectional_mean = returns_df.mean(axis=1)
    latest_features_dict = {}
    for col in prices_df.columns:
        if col == "^MXX":
            continue
        df = pd.DataFrame(index=prices_df.index)
        df['mom_5'] = prices_df[col].pct_change(5)
        df['mom_20'] = prices_df[col].pct_change(20)
        df['mom_60'] = prices_df[col].pct_change(60)
        df['vol_20'] = returns_df[col].rolling(20).std()
        df['zscore_20'] = (returns_df[col] - returns_df[col].rolling(20).mean()) / (df['vol_20'] + 1e-6)
        df['vol_shock'] = volumes_df[col] / (volumes_df[col].rolling(20).mean() + 1e-6)
        df['dispersion'] = returns_df[col] - cross_sectional_mean
        df['autocorr_20'] = returns_df[col].rolling(20).apply(lambda x: x.autocorr() if len(x) > 2 else 0, raw=False)
        df['target'] = returns_df[col].shift(-5).rolling(5).sum()
        df['asset'] = col
        latest_features_dict[col] = df.drop(columns=['target', 'asset']).iloc[-2]
        panel_data.append(df)
    df_all = pd.concat(panel_data)
    df_train = df_all.dropna(subset=['target']).iloc[:-5]
    if len(df_train) < 100:
        return pd.Series(0, index=[col for col in prices_df.columns if col != "^MXX"])
    X_train = df_train.drop(columns=['target', 'asset'])
    y_train = df_train['target']
    model = HistGradientBoostingRegressor(
        max_iter=50, max_depth=3, learning_rate=0.05,
        l2_regularization=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    X_today = pd.DataFrame(latest_features_dict).T
    preds = model.predict(X_today)
    ml_series = pd.Series(preds, index=X_today.index).rank(pct=True)
    if ml_series.std() > 0:
        return (ml_series - ml_series.mean()) / ml_series.std()
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
        try:
            model = sm.OLS(Y[:, i], X, missing='drop').fit()
            betas.append(model.params[1])
        except Exception:
            betas.append(1.0)
    return np.array(betas)


def detect_regime_gmm(market_returns):
    if len(market_returns) < 126:
        return "NEUTRAL"
    df_gmm = pd.DataFrame(index=market_returns.index)
    df_gmm['ret'] = market_returns.rolling(5).sum()
    df_gmm['vol'] = market_returns.rolling(20).std() * np.sqrt(252)
    df_gmm = df_gmm.dropna()
    if len(df_gmm) < 30:
        return "NEUTRAL"
    best_bic = np.inf
    best_gmm = None
    for n in [2, 3]:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(df_gmm.iloc[:-1])
        bic = gmm.bic(df_gmm.iloc[:-1])
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    state = best_gmm.predict(df_gmm.iloc[-1:])
    means = best_gmm.means_
    safe_state = np.argmin(means[:, 1])
    return "Mercado Estable" if state[0] == safe_state else "Mercado Volátil"


def calcular_drawdown_avanzado(retornos):
    if len(retornos) == 0:
        return pd.Series([0]), 0
    acumulado = (1 + retornos).cumprod()
    max_acum = acumulado.cummax()
    drawdown = (acumulado - max_acum) / max_acum
    return drawdown, drawdown.min()


def probabilistic_sharpe_ratio(port_returns, rf_daily, benchmark_sr=0.0):
    if len(port_returns) < 30:
        return 0, 0
    excess_ret = port_returns - rf_daily
    if np.isclose(excess_ret.std(), 0):
        return 0, 0
    sr_daily = excess_ret.mean() / excess_ret.std()
    sk = skew(port_returns)
    kt = kurtosis(port_returns)
    n = len(port_returns)
    sr_std = np.sqrt((1 - sk * sr_daily + ((kt - 1) / 4) * sr_daily**2) / (n - 1))
    psr = norm.cdf((sr_daily - (benchmark_sr / np.sqrt(252))) / sr_std) if sr_std > 0 else 0
    dsr_daily = sr_daily * (1 - (sk / 6) * sr_daily + ((kt - 3) / 24) * (sr_daily**2))
    dsr = dsr_daily * np.sqrt(252)
    return dsr, psr


# ==========================================
# 5. ROBUST OPTIMIZATION ENGINE
# ==========================================
def optimizar_market_neutral_pro(expected_returns, cov_matrix, current_weights, betas,
                                  adv_weights_max, vol_forecast_array, regime):
    num_assets = len(expected_returns)
    lambda_reg = 0.05 / num_assets
    rng = np.random.default_rng(seed=None)
    mc_scenarios = rng.multivariate_normal(np.zeros(num_assets), cov_matrix, 2000)
    lambda_cvar = 6.0 if regime == "Mercado Volátil" else 2.0

    def neg_sharpe(weights):
        p_ret = np.sum(expected_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        penalty_l2 = lambda_reg * np.sum(weights**2)
        delta = np.abs(weights - current_weights)
        spread_cost = 0.0005 * np.sum(delta)
        impact_cost = 0.005 * np.sum((delta**1.5) * vol_forecast_array / (adv_weights_max + 1e-6))
        sim_losses = -(mc_scenarios @ weights)
        sorted_losses = np.sort(sim_losses)
        cvar_true = np.mean(sorted_losses[-int(0.05 * 2000):])
        penalty_tail = lambda_cvar * max(cvar_true, 0)
        if np.isclose(p_vol, 0):
            return 0
        return -(p_ret) / p_vol + penalty_l2 + spread_cost + impact_cost + penalty_tail

    bounds = tuple((-mw, mw) for mw in adv_weights_max)
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w)},
        {'type': 'ineq', 'fun': lambda w: 0.05 - abs(np.dot(w, betas))},
        {'type': 'ineq', 'fun': lambda w: 1.5 - np.sum(np.abs(w))}
    )
    res = minimize(neg_sharpe, current_weights, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-9})
    if not res.success:
        st.warning(f"El optimizador necesitó más iteraciones: {res.message}. Se mantienen los pesos actuales por seguridad.")
        return current_weights
    return res.x


# ==========================================
# 6. WALK-FORWARD BACKTESTING
# ==========================================
def backtest_walk_forward_meta_model(prices, volumes, activos, rf_daily, initial_capital=1e6):
    dates = prices.index
    portfolio_returns = []
    bench_returns = []
    idx_out = []
    current_weights = np.zeros(len(activos))
    rebalance_freq = 21
    X_meta = []
    y_meta = []
    prev_stat = None
    prev_ml = None
    meta_model_final = None

    for i in range(252, len(dates) - rebalance_freq, rebalance_freq):
        start_idx = max(0, i - 504)
        train_prices = prices.iloc[start_idx:i]
        train_volumes = volumes.iloc[start_idx:i]
        train_returns = calcular_retornos_robustos(train_prices)
        test_slice = prices.iloc[i:i + rebalance_freq + 1]
        test_returns = test_slice.pct_change().dropna()
        if test_returns.empty or len(train_prices) < 126:
            continue

        if prev_stat is not None and prev_ml is not None:
            realized_per_asset = train_returns[activos].iloc[-rebalance_freq:].sum()
            for asset in activos:
                X_meta.append([prev_stat.get(asset, 0), prev_ml.get(asset, 0)])
                y_meta.append(realized_per_asset.get(asset, 0))

        c_stat = generate_multi_factor_alpha(train_prices[activos]).fillna(0)
        c_ml = generate_panel_ml_alpha(train_prices[activos], train_returns[activos], train_volumes[activos]).fillna(0)
        regime = detect_regime_gmm(train_returns["^MXX"])

        if len(X_meta) > len(activos) * 3:
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(X_meta, y_meta)
            meta_model_final = meta_model
            curr_X = np.column_stack([
                [c_stat.get(a, 0) for a in activos],
                [c_ml.get(a, 0) for a in activos]
            ])
            meta_alpha = pd.Series(meta_model.predict(curr_X), index=activos)
        else:
            w_ml = 0.2 if regime == "Mercado Volátil" else 0.7
            meta_alpha = ((1 - w_ml) * c_stat.reindex(activos).fillna(0)) + (w_ml * c_ml.reindex(activos).fillna(0))

        prev_stat = c_stat
        prev_ml = c_ml

        vol_forecast = train_returns[activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
        exp_ret = meta_alpha.values * vol_forecast.values
        cov = get_ledoit_wolf_cov(train_returns[activos])
        betas = calculate_betas(train_returns[activos], train_returns["^MXX"])
        adv_mxn = (train_prices[activos].tail(20) * train_volumes[activos].tail(20)).mean().values
        adv_weights_max = np.clip((0.10 * adv_mxn) / initial_capital, 0.01, 0.30)

        raw_weights = optimizar_market_neutral_pro(
            exp_ret, cov, current_weights, betas,
            adv_weights_max, vol_forecast.values, regime
        )
        current_vol = np.sqrt(np.dot(raw_weights.T, np.dot(cov, raw_weights)))
        max_dd_running = 0 if len(portfolio_returns) == 0 else calcular_drawdown_avanzado(pd.Series(portfolio_returns))[1]
        target_vol = 0.05 if (regime == "Mercado Volátil" or max_dd_running < -0.10) else 0.15
        new_weights = raw_weights * (target_vol / current_vol) if current_vol > 0 else raw_weights

        delta = np.abs(new_weights - current_weights)
        total_cost = (0.0005 * np.sum(delta)) + (0.005 * np.sum((delta**1.5) * vol_forecast.values / (adv_weights_max + 1e-6)))
        daily_exec_cost = total_cost / 5.0

        for j in range(len(test_returns)):
            day_ret = np.dot(new_weights, test_returns[activos].iloc[j].values)
            if j < 5:
                day_ret -= daily_exec_cost
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
# PLOTLY THEME HELPER
# ==========================================
def confidelis_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=WHITE)),
        plot_bgcolor=GRAY_900,
        paper_bgcolor=GRAY_800,
        font=dict(color=GRAY_200, family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=GRAY_300, size=11),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=GRAY_700, zerolinecolor=GRAY_700),
        yaxis=dict(gridcolor=GRAY_700, zerolinecolor=GRAY_700),
    )
    return fig


# ==========================================
# UI — SIDEBAR
# ==========================================
with st.sidebar:
    if logo_b64:
        st.markdown(f"""
            <div class='logo-container'>
                <img src="data:image/png;base64,{logo_b64}" alt="Confidelis"/>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:{TEAL_MID}; text-align:center;'>CONFIDELIS</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='text-align:center; font-size:0.78rem; color:{GRAY_400} !important;'>Gestión Patrimonial Inteligente</p>", unsafe_allow_html=True)
    st.markdown("")

    menu = st.radio(
        "Navegación",
        ["Mi Portafolio", "Simulación Histórica"],
        help="Seleccione la sección que desea consultar"
    )

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.82rem; color:{GRAY_300} !important; font-weight:500;'>Configuración</p>", unsafe_allow_html=True)
    rf_input = st.number_input(
        "Tasa libre de riesgo anual (%)",
        value=11.00, step=0.1,
        help="Es el rendimiento que ofrece un instrumento sin riesgo, como los CETES. Se usa como referencia para medir si tu portafolio genera valor adicional."
    )
    rf_daily = (rf_input / 100) / 252

    st.markdown("---")
    st.markdown(f"""
        <div style='text-align:center; padding: 10px;'>
            <p style='font-size:0.72rem; color:{GRAY_400} !important; margin:0;'>
                Datos: Yahoo Finance<br>
                Modelos: Machine Learning + Optimización<br>
                Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </p>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE: MI PORTAFOLIO (Live Execution Desk)
# ==========================================
if menu == "Mi Portafolio":
    st.markdown(f"<h1 style='color:{TEAL_LIGHT};'>Mi Portafolio</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='explainer'>Aquí puedes armar tu portafolio de inversión, ver cómo se comportan tus activos y recibir recomendaciones inteligentes de compra/venta para el día siguiente.</p>", unsafe_allow_html=True)

    # --- STEP 1: Add assets ---
    st.markdown(f"<h3>1. Arma tu portafolio</h3>", unsafe_allow_html=True)

    tab_manual, tab_popular = st.tabs(["Escribir ticker", "Activos populares"])

    with tab_manual:
        st.markdown(f"<p class='explainer'>Escribe el nombre corto (ticker) de la acción. Por ejemplo: WALMEX para Walmart de México, AMXB para América Móvil. Se agrega automáticamente el sufijo .MX</p>", unsafe_allow_html=True)
        with st.form("add_asset_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                nuevo_ticker = st.text_input("Ticker (ej. WALMEX, CEMEXCPO, BIMBOA)")
            with c2:
                weight_actual = st.number_input(
                    "Peso actual en tu portafolio (%)", value=0.0, step=1.0, min_value=0.0, max_value=100.0,
                    help="Si ya tienes este activo, indica qué porcentaje de tu capital representa. Si no lo tienes aún, déjalo en 0%."
                )
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                btn_agregar = st.form_submit_button("Agregar")
            if btn_agregar and nuevo_ticker:
                t_str = nuevo_ticker.strip().upper()
                if not t_str.endswith(".MX"):
                    t_str += ".MX"
                st.session_state['cartera'].append({"Ticker": t_str, "Peso": weight_actual / 100})
                st.rerun()

    with tab_popular:
        st.markdown(f"<p class='explainer'>Selecciona de la lista de acciones más operadas en la Bolsa Mexicana de Valores.</p>", unsafe_allow_html=True)
        selected_popular = st.multiselect(
            "Selecciona activos",
            options=list(POPULAR_MX_TICKERS.keys()),
            format_func=lambda x: f"{x.replace('.MX','')} — {POPULAR_MX_TICKERS[x]}",
        )
        if st.button("Agregar seleccionados") and selected_popular:
            for t in selected_popular:
                if not any(item['Ticker'] == t for item in st.session_state['cartera']):
                    st.session_state['cartera'].append({"Ticker": t, "Peso": 0.0})
            st.rerun()

    # --- SHOW CURRENT PORTFOLIO ---
    if st.session_state['cartera']:
        st.markdown(f"<h3>Tu portafolio actual ({len(st.session_state['cartera'])} activos)</h3>", unsafe_allow_html=True)

        pills_html = ""
        for item in st.session_state['cartera']:
            name = POPULAR_MX_TICKERS.get(item['Ticker'], item['Ticker'].replace('.MX', ''))
            peso_str = f" ({item['Peso']:.0%})" if item['Peso'] > 0 else ""
            pills_html += f"<span class='asset-pill'>{name}{peso_str}</span>"
        st.markdown(pills_html, unsafe_allow_html=True)

        if len(st.session_state['cartera']) < 5:
            remaining = 5 - len(st.session_state['cartera'])
            st.markdown(f"""
                <div class='alert-info'>
                    <b>Necesitas al menos 5 activos</b> para que el modelo de inteligencia artificial funcione correctamente.
                    Te faltan <b>{remaining}</b> más. Mientras más activos agregues, mejor podrá diversificar tu portafolio.
                </div>
            """, unsafe_allow_html=True)

        c_clear, _ = st.columns([1, 3])
        with c_clear:
            if st.button("Limpiar portafolio"):
                st.session_state['cartera'] = []
                st.rerun()

    # --- META MODEL STATUS ---
    if st.session_state['meta_model'] is not None:
        st.markdown(f"<span class='badge badge-ok'>Modelo de IA entrenado y listo</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge badge-warn'>Modelo de IA no entrenado. Ejecuta la Simulación Histórica primero para mayor precisión.</span>", unsafe_allow_html=True)

    st.markdown("---")

    # --- STEP 2: Asset details & analysis ---
    if len(st.session_state['cartera']) > 0:
        st.markdown(f"<h3>2. Estado de tus activos</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='explainer'>Descargamos los datos históricos de cada acción para mostrarte cómo se ha comportado. Esto te ayuda a entender qué tan riesgoso o rentable ha sido cada activo.</p>", unsafe_allow_html=True)

        activos_lista = list(set([item["Ticker"] for item in st.session_state['cartera']]))

        with st.spinner("Consultando datos del mercado..."):
            precios_preview, _ = fetch_market_data_cached(activos_lista, "2y")

        if precios_preview is not None:
            summaries = []
            for t in activos_lista:
                s = get_asset_summary(precios_preview, t)
                if s:
                    s["Activo"] = POPULAR_MX_TICKERS.get(t, t.replace('.MX', ''))
                    s["Ticker"] = t
                    summaries.append(s)
            if summaries:
                df_summary = pd.DataFrame(summaries)
                df_summary = df_summary[["Activo", "Ticker", "Precio", "Rend. 1 Mes", "Rend. 3 Meses", "Rend. 1 Año", "Volatilidad Anual", "Dist. desde Máximo"]]
                st.dataframe(df_summary.set_index("Activo"), use_container_width=True)

                st.markdown(f"""
                    <div class='info-card'>
                        <h4 style='color:{TEAL_LIGHT} !important;'>¿Cómo leer esta tabla?</h4>
                        <p>
                            <b>Rendimiento</b>: cuánto ha subido o bajado el precio en ese período. Un valor positivo (+) es ganancia, negativo (-) es pérdida.<br>
                            <b>Volatilidad Anual</b>: mide qué tan bruscos son los movimientos del precio. Mayor volatilidad = más riesgo pero también más oportunidad.<br>
                            <b>Dist. desde Máximo</b>: qué tan lejos está el precio actual de su punto más alto histórico. Si dice -15%, significa que ha caído 15% desde su mejor momento.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Mini chart of prices (normalized)
            chart_tickers = [t for t in activos_lista if t in precios_preview.columns and t != "^MXX"]
            if chart_tickers:
                normalized = precios_preview[chart_tickers].iloc[-252:].div(precios_preview[chart_tickers].iloc[-252:].iloc[0]) * 100
                fig_prices = go.Figure()
                colors_cycle = [TEAL_LIGHT, TEAL_GLOW, GRAY_300, WARN_SOFT, SUCCESS, "#9B8EC5", "#E8927C"]
                for idx, col in enumerate(normalized.columns):
                    display_name = POPULAR_MX_TICKERS.get(col, col.replace('.MX', ''))
                    fig_prices.add_trace(go.Scatter(
                        x=normalized.index, y=normalized[col],
                        name=display_name,
                        line=dict(color=colors_cycle[idx % len(colors_cycle)], width=2)
                    ))
                confidelis_layout(fig_prices, "Evolución del último año (base 100)")
                fig_prices.update_yaxes(title_text="Valor relativo")
                st.plotly_chart(fig_prices, use_container_width=True)
                st.markdown(f"<p class='explainer'>Esta gráfica compara cómo se han movido tus activos en el último año. Todos parten de 100 para que puedas comparar fácilmente. Si una línea está en 120, significa que subió 20%.</p>", unsafe_allow_html=True)

    # --- STEP 3: Generate orders ---
    if len(st.session_state['cartera']) > 4:
        st.markdown("---")
        st.markdown(f"<h3>3. Recomendaciones de inversión</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='explainer'>Nuestro sistema analiza tendencias, volatilidad y señales del mercado usando inteligencia artificial para sugerirte qué comprar, vender o mantener para mañana.</p>", unsafe_allow_html=True)

        df_cart = pd.DataFrame(st.session_state['cartera']).groupby("Ticker").sum().reset_index()
        activos_brutos = df_cart["Ticker"].tolist()
        current_weights_dict = dict(zip(df_cart["Ticker"], df_cart["Peso"]))
        capital_operativo = st.number_input(
            "¿Cuánto capital tienes invertido? ($ MXN)",
            value=1000000, step=100000, min_value=100000,
            help="Ingresa el monto total de dinero que tienes disponible para invertir. El sistema calculará cuánto mover en pesos."
        )

        if st.button("Generar recomendaciones para mañana"):
            with st.spinner("Analizando el mercado con inteligencia artificial... esto puede tomar unos segundos"):
                precios_historicos, volumen_historico = fetch_market_data(activos_brutos, "5y")

                if precios_historicos is None:
                    st.error("No se pudo conectar con el proveedor de datos. Verifica tu conexión a internet e intenta de nuevo.")
                    st.stop()

                valid_tickers = [col for col in precios_historicos.columns if precios_historicos[col].count() >= 252 or col == "^MXX"]
                excluidos = set(activos_brutos) - set(valid_tickers)
                if excluidos:
                    nombres_excl = [POPULAR_MX_TICKERS.get(t, t) for t in excluidos]
                    st.warning(f"Se excluyeron por no tener suficiente historial (se necesita al menos 1 año): {', '.join(nombres_excl)}")

                activos = [a for a in activos_brutos if a in valid_tickers]
                if len(activos) < 5:
                    st.error("No hay suficientes activos con historial. Agrega más activos a tu portafolio.")
                    st.stop()

                current_weights = np.array([current_weights_dict.get(a, 0) for a in activos])
                returns = calcular_retornos_robustos(precios_historicos)

                if len(precios_historicos) > 504:
                    train_prices = precios_historicos.iloc[-504:]
                    train_returns = returns.iloc[-504:]
                    train_volumes = volumen_historico.iloc[-504:]
                else:
                    train_prices = precios_historicos
                    train_returns = returns
                    train_volumes = volumen_historico

                alpha_stat = generate_multi_factor_alpha(train_prices[activos]).fillna(0)
                alpha_ml = generate_panel_ml_alpha(train_prices[activos], train_returns[activos], train_volumes[activos]).fillna(0)
                regime = detect_regime_gmm(train_returns["^MXX"])

                if st.session_state['meta_model'] is not None:
                    curr_X = np.column_stack([
                        [alpha_stat.get(a, 0) for a in activos],
                        [alpha_ml.get(a, 0) for a in activos]
                    ])
                    meta_alpha = pd.Series(st.session_state['meta_model'].predict(curr_X), index=activos)
                else:
                    w_ml = 0.2 if regime == "Mercado Volátil" else 0.7
                    meta_alpha = ((1 - w_ml) * alpha_stat.reindex(activos).fillna(0)) + (w_ml * alpha_ml.reindex(activos).fillna(0))

                vol_forecast = train_returns[activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
                exp_ret = meta_alpha.values * vol_forecast.values
                cov = get_ledoit_wolf_cov(train_returns[activos])
                betas = calculate_betas(train_returns[activos], train_returns["^MXX"])
                adv_mxn = (train_prices[activos].tail(20) * train_volumes[activos].tail(20)).mean().values
                adv_weights_max = np.clip((0.10 * adv_mxn) / capital_operativo, 0.01, 0.30)
                target_vol = 0.08 if regime == "Mercado Volátil" else 0.15

                raw_weights = optimizar_market_neutral_pro(
                    exp_ret, cov, current_weights, betas,
                    adv_weights_max, vol_forecast.values, regime
                )
                current_vol = np.sqrt(np.dot(raw_weights.T, np.dot(cov, raw_weights)))
                new_weights = raw_weights * (target_vol / current_vol) if current_vol > 0 else raw_weights

                # --- REGIME ALERT ---
                if regime == "Mercado Volátil":
                    st.markdown(f"""
                        <div class='alert-crisis'>
                            <b>El mercado se encuentra en fase volátil.</b><br>
                            Nuestro sistema detectó inestabilidad en el mercado mexicano. Por precaución, se ha reducido automáticamente
                            el nivel de riesgo de las recomendaciones para proteger tu patrimonio. En estos momentos es más importante
                            conservar que arriesgar.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='alert-stable'>
                            <b>El mercado se encuentra estable.</b><br>
                            Las condiciones actuales permiten operar con normalidad. El sistema optimiza buscando el mejor balance entre
                            rendimiento y protección.
                        </div>
                    """, unsafe_allow_html=True)

                # --- METRICS ---
                st.markdown(f"<h4>Resumen del portafolio recomendado</h4>", unsafe_allow_html=True)
                beta_neta = np.dot(new_weights, betas)
                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("Riesgo esperado (anual)", f"{current_vol:.1%}",
                           help="La volatilidad anualizada mide cuánto podrían variar tus rendimientos. Un 10% significa que tu portafolio podría subir o bajar aproximadamente esa cantidad en un año.")
                cm2.metric("Sensibilidad al mercado", f"{beta_neta:.3f}",
                           help="Beta mide cuánto se mueve tu portafolio cuando el mercado sube o baja. Un valor cercano a 0 significa que tu portafolio es independiente del mercado (market-neutral).")
                cm3.metric("Condición del mercado", regime)

                # --- ORDERS TABLE ---
                st.markdown(f"<h4>Movimientos sugeridos</h4>", unsafe_allow_html=True)
                st.markdown(f"<p class='explainer'>Esta tabla muestra qué hacer con cada activo. 'Comprar' significa aumentar tu posición, 'Reducir' significa vender parte, y 'Mantener' significa no hacer cambios significativos.</p>", unsafe_allow_html=True)

                df_orders = pd.DataFrame({
                    "Ticker": activos,
                    "Activo": [POPULAR_MX_TICKERS.get(a, a.replace('.MX', '')) for a in activos],
                    "Peso Actual": current_weights,
                    "Peso Sugerido": new_weights
                })
                df_orders["Cambio"] = df_orders["Peso Sugerido"] - df_orders["Peso Actual"]
                df_orders["Monto a mover ($)"] = df_orders["Cambio"] * capital_operativo

                def classify_action(delta):
                    if delta > 0.01:
                        return "Comprar"
                    elif delta < -0.01:
                        return "Reducir"
                    else:
                        return "Mantener"

                df_orders["Acción"] = df_orders["Cambio"].apply(classify_action)

                # Format for display
                df_display = df_orders.copy()
                df_display["Peso Actual"] = df_display["Peso Actual"].apply(lambda x: f"{x:.1%}")
                df_display["Peso Sugerido"] = df_display["Peso Sugerido"].apply(lambda x: f"{x:.1%}")
                df_display["Cambio"] = df_display["Cambio"].apply(lambda x: f"{x:+.1%}")
                df_display["Monto a mover ($)"] = df_display["Monto a mover ($)"].apply(lambda x: f"${x:+,.0f}")
                df_display = df_display[["Activo", "Ticker", "Peso Actual", "Peso Sugerido", "Cambio", "Monto a mover ($)", "Acción"]]

                st.dataframe(df_display.set_index("Activo").sort_values(by="Acción"), use_container_width=True)

                # --- WEIGHT COMPARISON CHART ---
                fig_weights = go.Figure()
                display_names = [POPULAR_MX_TICKERS.get(a, a.replace('.MX', '')) for a in activos]
                fig_weights.add_trace(go.Bar(
                    name="Peso Actual", x=display_names, y=df_orders["Peso Actual"].values,
                    marker_color=GRAY_400
                ))
                fig_weights.add_trace(go.Bar(
                    name="Peso Sugerido", x=display_names, y=df_orders["Peso Sugerido"].values,
                    marker_color=TEAL_MID
                ))
                confidelis_layout(fig_weights, "Comparación: tu portafolio actual vs. el recomendado")
                fig_weights.update_layout(barmode='group')
                fig_weights.update_yaxes(tickformat='.0%', title_text="Peso en portafolio")
                st.plotly_chart(fig_weights, use_container_width=True)
                st.markdown(f"<p class='explainer'>Las barras grises muestran cómo está distribuido tu portafolio hoy. Las barras en teal muestran cómo quedaría si sigues las recomendaciones del sistema. Valores negativos significan posiciones 'en corto' (apostar a que baje).</p>", unsafe_allow_html=True)

                csv = df_display.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="Descargar recomendaciones (CSV)",
                    data=csv,
                    file_name=f'confidelis_recomendaciones_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

    elif len(st.session_state['cartera']) > 0:
        st.markdown(f"""
            <div class='alert-info'>
                <b>Agrega más activos para activar las recomendaciones.</b><br>
                Necesitas al menos 5 activos en tu portafolio. Actualmente tienes {len(st.session_state['cartera'])}.
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE: SIMULACION HISTORICA (Backtest)
# ==========================================
elif menu == "Simulación Histórica":
    st.markdown(f"<h1 style='color:{TEAL_LIGHT};'>Simulación Histórica</h1>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class='info-card'>
            <h4 style='color:{TEAL_LIGHT} !important;'>¿Qué es esto?</h4>
            <p>
                Esta herramienta prueba cómo habría funcionado nuestra estrategia de inversión <b>en el pasado</b>,
                usando datos reales del mercado mexicano. Es como una "máquina del tiempo financiera" que nos permite
                evaluar si el modelo hubiera generado ganancias.<br><br>
                <b>Importante:</b> Los resultados pasados <b>no garantizan</b> resultados futuros, pero son la mejor
                forma de validar que una estrategia tiene fundamento sólido antes de usarla con dinero real.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if len(st.session_state['cartera']) > 4:
        activos_brutos = list(set([item["Ticker"] for item in st.session_state['cartera']]))

        # Show what will be tested
        st.markdown(f"<h4>Activos que se analizarán:</h4>", unsafe_allow_html=True)
        pills_html = ""
        for t in activos_brutos:
            name = POPULAR_MX_TICKERS.get(t, t.replace('.MX', ''))
            pills_html += f"<span class='asset-pill'>{name}</span>"
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("")

        st.markdown(f"""
            <div class='info-card'>
                <h4 style='color:{TEAL_LIGHT} !important;'>¿Qué hace el motor de simulación?</h4>
                <p>
                    1. Toma los últimos 5 años de datos del mercado<br>
                    2. Cada mes, el modelo de IA analiza tendencias y decide cómo repartir el capital<br>
                    3. Incluye costos reales de transacción (comisiones, impacto de mercado)<br>
                    4. Compara los resultados contra el IPC (índice principal de la Bolsa Mexicana)<br>
                    5. Entrena un modelo predictivo que luego se usa en las recomendaciones en vivo
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Ejecutar simulación completa"):
            progress_bar = st.progress(0, text="Iniciando...")

            progress_bar.progress(10, text="Descargando datos históricos del mercado...")
            precios_historicos, volumen_historico = fetch_market_data_cached(activos_brutos, "5y")

            if precios_historicos is None:
                st.error("No se pudieron obtener datos históricos. Verifica tu conexión e intenta de nuevo.")
                st.stop()

            progress_bar.progress(25, text="Validando activos...")
            valid_tickers = [col for col in precios_historicos.columns if precios_historicos[col].count() >= 252 or col == "^MXX"]
            activos = [a for a in activos_brutos if a in valid_tickers]

            if len(activos) < 5:
                st.error("No hay suficientes activos con historial de al menos 1 año.")
                st.stop()

            excluidos = set(activos_brutos) - set(activos)
            if excluidos:
                nombres_excl = [POPULAR_MX_TICKERS.get(t, t) for t in excluidos]
                st.warning(f"Excluidos por falta de historial: {', '.join(nombres_excl)}")

            progress_bar.progress(40, text="Entrenando modelos de inteligencia artificial... (esto puede tomar un momento)")

            oos_ret, trained_meta_model, bench_ret = backtest_walk_forward_meta_model(
                precios_historicos, volumen_historico, activos, rf_daily, initial_capital=10000000
            )

            progress_bar.progress(90, text="Calculando métricas y generando gráficas...")

            if trained_meta_model is not None:
                st.session_state['meta_model'] = trained_meta_model

            if len(oos_ret) > 0:
                total_ret = np.prod(1 + oos_ret) - 1
                ann_ret = (1 + total_ret) ** (252 / len(oos_ret)) - 1
                vol = oos_ret.std() * np.sqrt(252)
                dsr, psr = probabilistic_sharpe_ratio(oos_ret, rf_daily)
                sharpe = (ann_ret - (rf_daily * 252)) / vol if vol > 0 else 0
                drawdown, max_dd = calcular_drawdown_avanzado(oos_ret)

                progress_bar.progress(100, text="Listo")

                # --- RESULTS ---
                st.markdown("---")
                st.markdown(f"<h3>Resultados de la simulación</h3>", unsafe_allow_html=True)

                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric(
                    "Rendimiento anual",
                    f"{ann_ret:.1%}",
                    help="Rendimiento compuesto anualizado que habría generado la estrategia. Es como decir 'en promedio, cada año habría ganado este porcentaje'."
                )
                cm2.metric(
                    "Riesgo (volatilidad)",
                    f"{vol:.1%}",
                    help="Qué tanto variaron los rendimientos. Menor es mejor porque significa más estabilidad."
                )
                cm3.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    help="Mide cuánto rendimiento obtuviste por cada unidad de riesgo. Mayor a 1.0 se considera bueno. Mayor a 2.0 es excelente."
                )
                cm4.metric(
                    "Peor caída",
                    f"{max_dd:.1%}",
                    help="La mayor pérdida acumulada desde un punto alto. Por ejemplo, -15% significa que en su peor momento, el portafolio habría caído 15% antes de recuperarse."
                )

                cm5, cm6 = st.columns(2)
                cm5.metric(
                    "Confianza estadística (PSR)",
                    f"{psr:.0%}",
                    help="Probabilistic Sharpe Ratio: la probabilidad de que el rendimiento no sea producto del azar. Mayor a 95% es un resultado estadísticamente significativo."
                )
                cm6.metric(
                    "Sharpe ajustado (DSR)",
                    f"{dsr:.2f}",
                    help="Deflated Sharpe Ratio: una versión más conservadora del Sharpe que penaliza la asimetría y las colas pesadas de los rendimientos."
                )

                # Explain results
                if sharpe > 1.5 and psr > 0.90:
                    st.markdown(f"""
                        <div class='alert-stable'>
                            <b>Resultado sólido.</b> La estrategia muestra rendimientos consistentes con alta confianza estadística.
                            El modelo de IA ha sido entrenado y está disponible para generar recomendaciones en la sección "Mi Portafolio".
                        </div>
                    """, unsafe_allow_html=True)
                elif sharpe > 0.5:
                    st.markdown(f"""
                        <div class='alert-info'>
                            <b>Resultado aceptable.</b> La estrategia genera valor, pero con margen moderado.
                            Considera diversificar más tu portafolio para mejorar los resultados.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='alert-crisis'>
                            <b>Resultado débil.</b> Con estos activos, la estrategia no genera suficiente valor.
                            Prueba con un universo de activos diferente o más diversificado.
                        </div>
                    """, unsafe_allow_html=True)

                # --- EQUITY CHART ---
                equity = (1 + oos_ret).cumprod()
                equity_bench = (1 + bench_ret).cumprod()

                fig_oos = go.Figure()

                shuffle_paths = monte_carlo_shuffle_test(oos_ret, n_sim=50)
                for path in shuffle_paths:
                    path_len = len(path)
                    fig_oos.add_trace(go.Scatter(
                        x=equity.index[:path_len], y=path,
                        mode='lines', line=dict(color=GRAY_600, width=1),
                        opacity=0.15, showlegend=False
                    ))

                fig_oos.add_trace(go.Scatter(
                    x=equity.index, y=equity,
                    name="Estrategia Confidelis",
                    line=dict(color=TEAL_LIGHT, width=3)
                ))
                fig_oos.add_trace(go.Scatter(
                    x=equity_bench.index, y=equity_bench,
                    name="Bolsa Mexicana (IPC)",
                    line=dict(color=GRAY_400, width=2, dash='dot')
                ))
                confidelis_layout(fig_oos, "Crecimiento de $1 invertido (estrategia vs. mercado)")
                st.plotly_chart(fig_oos, use_container_width=True)

                st.markdown(f"""
                    <div class='info-card'>
                        <h4 style='color:{TEAL_LIGHT} !important;'>¿Cómo leer esta gráfica?</h4>
                        <p>
                            La <b>línea teal</b> muestra cómo habría crecido tu dinero con la estrategia de Confidelis.
                            La <b>línea gris punteada</b> muestra lo que habría pasado si simplemente comprabas todo el mercado (IPC).
                            Las <b>líneas grises tenues</b> son simulaciones aleatorias que demuestran que los resultados no son producto del azar.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- DRAWDOWN CHART ---
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown,
                    fill='tozeroy', name="Caída acumulada",
                    line=dict(color=DANGER, width=2),
                    fillcolor=f"rgba(176,64,80,0.15)"
                ))
                confidelis_layout(fig_dd, "Caídas del portafolio (drawdown)")
                fig_dd.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig_dd, use_container_width=True)

                st.markdown(f"<p class='explainer'>Esta gráfica muestra los momentos en que el portafolio perdió valor antes de recuperarse. Es normal que haya caídas temporales; lo importante es que se recupere. La profundidad máxima fue de {max_dd:.1%}.</p>", unsafe_allow_html=True)

            else:
                progress_bar.progress(100, text="Completado")
                st.warning("No hubo suficientes datos para completar la simulación. Intenta con activos que tengan más historial.")
    else:
        n_actual = len(st.session_state['cartera'])
        st.markdown(f"""
            <div class='alert-info'>
                <b>Primero arma tu portafolio.</b><br>
                Ve a la sección "Mi Portafolio" y agrega al menos 5 activos.
                {"Actualmente tienes " + str(n_actual) + "." if n_actual > 0 else "Aún no has agregado ningún activo."}
            </div>
        """, unsafe_allow_html=True)

