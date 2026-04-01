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
import re

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
    page_title="CONFIDELIS - Gestión Patrimonial",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta Confidelis — fondo claro, acentos teal
# Se evitan: naranja #FF6900 y azul fuerte #13294B
TEAL_DARK = "#2B6E6E"
TEAL_MID = "#3D8C8C"
TEAL_LIGHT = "#5AACAC"
TEAL_GLOW = "#7ECFCF"
BG_LIGHT = "#F5F7FA"
BG_WHITE = "#FFFFFF"
BG_CARD = "#FFFFFF"
BORDER = "#E2E8F0"
GRAY_700 = "#374151"
GRAY_600 = "#4B5563"
GRAY_500 = "#6B7280"
GRAY_400 = "#9CA3AF"
GRAY_300 = "#D1D5DB"
TEXT_DARK = "#1F2937"
TEXT_MID = "#4B5563"
SUCCESS = "#2E8B57"
DANGER = "#B04050"
WARN_SOFT = "#C9A030"

LOGO_URL = "https://raw.githubusercontent.com/confidelis-mx/assets/main/logo_confidelis.png"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background-color: {BG_LIGHT};
        color: {TEXT_DARK};
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT_DARK} !important; font-weight: 600; letter-spacing: -0.02em; }}
    p, span, label, li {{ color: {TEXT_MID} !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {BG_WHITE} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{ color: {TEXT_MID} !important; }}

    hr {{ border-color: {BORDER}; }}

    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {{
        background-color: {BG_WHITE} !important;
        color: {TEXT_DARK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px;
    }}
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
        border-color: {TEAL_MID} !important;
        box-shadow: 0 0 0 3px rgba(61,140,140,0.15);
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
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {TEAL_MID}, {TEAL_LIGHT}) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(61,140,140,0.25);
    }}

    /* Metrics */
    div[data-testid="metric-container"] {{
        background-color: {BG_WHITE};
        border: 1px solid {BORDER};
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEAL_DARK} !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {GRAY_500} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {BG_WHITE} !important;
        border-radius: 10px;
        color: {TEXT_DARK} !important;
    }}

    /* DataFrames */
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

    /* Custom cards */
    .info-card {{
        background: {BG_WHITE};
        border: 1px solid {BORDER};
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .info-card h4 {{ margin: 0 0 8px 0; font-size: 1rem; color: {TEAL_DARK} !important; }}
    .info-card p {{ margin: 0; font-size: 0.9rem; line-height: 1.6; }}

    .alert-crisis {{
        background: rgba(176,64,80,0.06);
        border-left: 4px solid {DANGER};
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
    }}
    .alert-stable {{
        background: rgba(46,139,87,0.06);
        border-left: 4px solid {SUCCESS};
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
    }}
    .alert-info {{
        background: rgba(61,140,140,0.06);
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
    }}
    .badge-ok {{ background-color: rgba(46,139,87,0.1); color: {SUCCESS}; border: 1px solid rgba(46,139,87,0.2); }}
    .badge-warn {{ background-color: rgba(201,160,48,0.1); color: {WARN_SOFT}; border: 1px solid rgba(201,160,48,0.2); }}

    .asset-pill {{
        display: inline-block;
        background: {BG_LIGHT};
        border: 1px solid {BORDER};
        color: {TEAL_DARK} !important;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    .explainer {{
        color: {GRAY_500} !important;
        font-size: 0.82rem;
        font-style: italic;
        margin-top: 4px;
        line-height: 1.5;
    }}

    .logo-container {{
        text-align: center;
        padding: 16px 10px 8px 10px;
    }}
    .logo-container img {{ max-width: 180px; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_WHITE};
        border-radius: 8px;
        color: {GRAY_500} !important;
        border: 1px solid {BORDER};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {TEAL_DARK} !important;
        color: white !important;
    }}

    .stDownloadButton>button {{
        background: {BG_WHITE} !important;
        border: 1px solid {BORDER} !important;
        color: {TEAL_DARK} !important;
    }}
    .stDownloadButton>button:hover {{
        background: {BG_LIGHT} !important;
        border-color: {TEAL_MID} !important;
    }}
    </style>
""", unsafe_allow_html=True)

if 'cartera' not in st.session_state:
    st.session_state['cartera'] = []
if 'meta_model' not in st.session_state:
    st.session_state['meta_model'] = None

# ==========================================
# 2. DATA ENGINE — CETES + MARKET DATA
# ==========================================
POPULAR_MX_TICKERS = {
    "WALMEX.MX": "Walmart de México",
    "AMXB.MX": "América Móvil",
    "FEMSAUBD.MX": "FEMSA",
    "GMEXICOB.MX": "Grupo México",
    "GFNORTEO.MX": "Banorte",
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
    "BOLSAA.MX": "Bolsa Mexicana de Valores",
}


@st.cache_data(ttl=86400)
def fetch_cetes_rate():
    """Obtiene la tasa CETES 28 días más reciente desde Banxico."""
    try:
        import requests as req_lib
        from bs4 import BeautifulSoup
        r = req_lib.get(
            'https://www.banxico.org.mx/SieInternet/consultarDirectorioInternetAction.do'
            '?sector=22&accion=consultarCuadro&idCuadro=CF107&locale=es',
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=15
        )
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) == 2:
                cells = [td.get_text(strip=True) for td in rows[1].find_all(['td', 'th'])]
                vals = []
                for c in cells:
                    if re.match(r'^\d+\.\d+$', c):
                        v = float(c)
                        if 3.0 < v < 15.0:
                            vals.append(v)
                if vals:
                    return vals[-1]
    except Exception:
        pass
    return None


def fetch_market_data(tickers_list, period="5y"):
    """Descarga precios y volúmenes de Yahoo Finance."""
    if not tickers_list:
        return None, None
    tickers = list(set(tickers_list))
    if "^MXX" not in tickers:
        tickers.append("^MXX")
    try:
        data = yf.download(tickers, period=period, progress=False)
    except Exception as e:
        st.error(f"No se pudo conectar con el proveedor de datos: {e}")
        return None, None
    if data.empty:
        return None, None

    # yfinance devuelve MultiIndex (Price, Ticker) para múltiples tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else None
        volumes = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
        if prices is None:
            return None, None
        if volumes is None:
            volumes = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
        # Limpiar nombres de columnas (quitar nivel 'Ticker' si existe)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.droplevel(0)
        if isinstance(volumes.columns, pd.MultiIndex):
            volumes.columns = volumes.columns.droplevel(0)
    else:
        # Un solo ticker
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
            volumes = data[['Volume']].rename(columns={'Volume': tickers[0]}) if 'Volume' in data.columns else pd.DataFrame(1e6, index=data.index, columns=[tickers[0]])
        else:
            return None, None

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No se encontraron datos para: {', '.join(missing)}")

    return prices.ffill().bfill(), volumes.ffill().bfill()


@st.cache_data(ttl=3600)
def fetch_market_data_cached(tickers_tuple, period="5y"):
    return fetch_market_data(list(tickers_tuple), period)


def calcular_retornos_robustos(prices, lookback_for_quantile=252):
    """Winsoriza retornos usando SOLO datos pasados (sin look-ahead bias)."""
    returns = prices.pct_change().dropna()
    if len(returns) < lookback_for_quantile:
        lookback_for_quantile = max(30, len(returns))
    lower = returns.rolling(lookback_for_quantile, min_periods=30).quantile(0.01)
    upper = returns.rolling(lookback_for_quantile, min_periods=30).quantile(0.99)
    clipped = returns.clip(lower=lower, upper=upper)
    return clipped.fillna(returns).fillna(0)


def get_asset_summary(prices_df, ticker):
    """Genera resumen de un activo."""
    if ticker not in prices_df.columns or len(prices_df[ticker].dropna()) < 20:
        return None
    p = prices_df[ticker].dropna()
    last_price = p.iloc[-1]
    ret_1m = (p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0
    ret_3m = (p.iloc[-1] / p.iloc[-63] - 1) if len(p) >= 63 else 0
    ret_1y = (p.iloc[-1] / p.iloc[-252] - 1) if len(p) >= 252 else 0
    vol = p.pct_change().dropna().std() * np.sqrt(252)
    dist_from_max = (last_price / p.max() - 1)
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
        signals[col] = (0.35 * rank_mom.get(col, 0) + 0.25 * vol_adj_mom.get(col, 0) +
                        0.15 * trend_factor.get(col, 0) + 0.15 * low_vol_factor.get(col, 0) + 0.10 * (-z_score / 3))
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
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.05, l2_regularization=0.1, random_state=42)
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
        return "Neutral"
    df_gmm = pd.DataFrame(index=market_returns.index)
    df_gmm['ret'] = market_returns.rolling(5).sum()
    df_gmm['vol'] = market_returns.rolling(20).std() * np.sqrt(252)
    df_gmm = df_gmm.dropna()
    if len(df_gmm) < 30:
        return "Neutral"
    best_bic, best_gmm = np.inf, None
    for n in [2, 3]:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(df_gmm.iloc[:-1])
        bic = gmm.bic(df_gmm.iloc[:-1])
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm
    state = best_gmm.predict(df_gmm.iloc[-1:])
    safe_state = np.argmin(best_gmm.means_[:, 1])
    return "Estable" if state[0] == safe_state else "Volátil"


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
    sk, kt, n = skew(port_returns), kurtosis(port_returns), len(port_returns)
    sr_std = np.sqrt((1 - sk * sr_daily + ((kt - 1) / 4) * sr_daily**2) / (n - 1))
    psr = norm.cdf((sr_daily - (benchmark_sr / np.sqrt(252))) / sr_std) if sr_std > 0 else 0
    dsr_daily = sr_daily * (1 - (sk / 6) * sr_daily + ((kt - 3) / 24) * (sr_daily**2))
    return dsr_daily * np.sqrt(252), psr


# ==========================================
# 5. OPTIMIZER
# ==========================================
def optimizar_market_neutral_pro(expected_returns, cov_matrix, current_weights, betas, adv_weights_max, vol_forecast_array, regime):
    num_assets = len(expected_returns)
    lambda_reg = 0.05 / num_assets
    rng = np.random.default_rng()
    mc_scenarios = rng.multivariate_normal(np.zeros(num_assets), cov_matrix, 2000)
    lambda_cvar = 6.0 if regime == "Volátil" else 2.0

    def neg_sharpe(weights):
        p_ret = np.sum(expected_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        penalty_l2 = lambda_reg * np.sum(weights**2)
        delta = np.abs(weights - current_weights)
        spread_cost = 0.0005 * np.sum(delta)
        impact_cost = 0.005 * np.sum((delta**1.5) * vol_forecast_array / (adv_weights_max + 1e-6))
        sim_losses = -(mc_scenarios @ weights)
        cvar_true = np.mean(np.sort(sim_losses)[-100:])
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
    res = minimize(neg_sharpe, current_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
    return res.x if res.success else current_weights


# ==========================================
# 6. BACKTEST
# ==========================================
def backtest_walk_forward_meta_model(prices, volumes, activos, rf_daily, initial_capital=1e6):
    dates = prices.index
    portfolio_returns, bench_returns, idx_out = [], [], []
    current_weights = np.zeros(len(activos))
    rebalance_freq = 21
    X_meta, y_meta = [], []
    prev_stat, prev_ml = None, None
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

        # Verificar que los activos existan en train_returns
        valid_activos = [a for a in activos if a in train_returns.columns and a in test_returns.columns]
        if len(valid_activos) < 5 or "^MXX" not in train_returns.columns or "^MXX" not in test_returns.columns:
            continue

        if prev_stat is not None and prev_ml is not None:
            realized = train_returns[valid_activos].iloc[-rebalance_freq:].sum()
            for asset in valid_activos:
                X_meta.append([prev_stat.get(asset, 0), prev_ml.get(asset, 0)])
                y_meta.append(realized.get(asset, 0))

        c_stat = generate_multi_factor_alpha(train_prices[valid_activos]).fillna(0)
        c_ml = generate_panel_ml_alpha(train_prices[valid_activos], train_returns[valid_activos], train_volumes[valid_activos]).fillna(0)
        regime = detect_regime_gmm(train_returns["^MXX"])

        if len(X_meta) > len(valid_activos) * 3:
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(X_meta, y_meta)
            meta_model_final = meta_model
            curr_X = np.column_stack([[c_stat.get(a, 0) for a in valid_activos], [c_ml.get(a, 0) for a in valid_activos]])
            meta_alpha = pd.Series(meta_model.predict(curr_X), index=valid_activos)
        else:
            w_ml = 0.2 if regime == "Volátil" else 0.7
            meta_alpha = ((1 - w_ml) * c_stat.reindex(valid_activos).fillna(0)) + (w_ml * c_ml.reindex(valid_activos).fillna(0))

        prev_stat, prev_ml = c_stat, c_ml

        vol_forecast = train_returns[valid_activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
        exp_ret = meta_alpha.values * vol_forecast.values
        cov = get_ledoit_wolf_cov(train_returns[valid_activos])
        betas = calculate_betas(train_returns[valid_activos], train_returns["^MXX"])
        adv_mxn = (train_prices[valid_activos].tail(20) * train_volumes[valid_activos].tail(20)).mean().values
        adv_weights_max = np.clip((0.10 * adv_mxn) / initial_capital, 0.01, 0.30)

        # Ajustar current_weights al tamaño de valid_activos si cambió
        if len(current_weights) != len(valid_activos):
            current_weights = np.zeros(len(valid_activos))

        raw_weights = optimizar_market_neutral_pro(exp_ret, cov, current_weights, betas, adv_weights_max, vol_forecast.values, regime)
        current_vol = np.sqrt(np.dot(raw_weights.T, np.dot(cov, raw_weights)))
        max_dd_running = 0 if not portfolio_returns else calcular_drawdown_avanzado(pd.Series(portfolio_returns))[1]
        target_vol = 0.05 if (regime == "Volátil" or max_dd_running < -0.10) else 0.15
        new_weights = raw_weights * (target_vol / current_vol) if current_vol > 0 else raw_weights

        delta = np.abs(new_weights - current_weights)
        total_cost = (0.0005 * np.sum(delta)) + (0.005 * np.sum((delta**1.5) * vol_forecast.values / (adv_weights_max + 1e-6)))
        daily_exec_cost = total_cost / 5.0

        for j in range(len(test_returns)):
            day_ret = np.dot(new_weights, test_returns[valid_activos].iloc[j].values)
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
        results.append((1 + shuffled).cumprod().values)
    return results


# ==========================================
# PLOTLY THEME
# ==========================================
def confidelis_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_DARK)),
        plot_bgcolor=BG_WHITE,
        paper_bgcolor=BG_WHITE,
        font=dict(color=TEXT_MID, family="Inter, sans-serif", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=GRAY_600, size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=GRAY_300, zerolinecolor=GRAY_300),
        yaxis=dict(gridcolor=GRAY_300, zerolinecolor=GRAY_300),
    )
    return fig


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(f"<div class='logo-container'><h2 style='color:{TEAL_DARK}; margin:0;'>CONFIDELIS</h2><p style='font-size:0.75rem; color:{GRAY_500} !important; margin:0;'>El aliado que tu patrimonio merece</p></div>", unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio("Navegación", ["Mi Portafolio", "Análisis de Inversión", "Simulación Histórica"])

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.82rem; color:{GRAY_600} !important; font-weight:500;'>Tasa de Referencia</p>", unsafe_allow_html=True)

    cetes_auto = fetch_cetes_rate()
    if cetes_auto:
        st.markdown(f"<div class='alert-info' style='padding:10px 14px; margin:0 0 8px 0;'><b>CETES 28 días (Banxico):</b> {cetes_auto:.2f}%</div>", unsafe_allow_html=True)
        rf_default = cetes_auto
    else:
        rf_default = 9.00

    rf_input = st.number_input(
        "Tasa libre de riesgo anual (%)", value=rf_default, step=0.1,
        help="Tasa CETES vigente. Se obtiene automáticamente de Banxico. Puedes ajustarla manualmente."
    )
    rf_daily = (rf_input / 100) / 252

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.7rem; color:{GRAY_400} !important; text-align:center;'>Datos: Yahoo Finance + Banxico<br>{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>", unsafe_allow_html=True)


# ==========================================
# HELPER: add assets UI
# ==========================================
def render_add_assets():
    tab_manual, tab_popular = st.tabs(["Escribir ticker", "Activos populares BMV"])
    with tab_manual:
        st.markdown(f"<p class='explainer'>Escribe el ticker de la acción (ej. WALMEX, CEMEXCPO). Se agrega .MX automáticamente.</p>", unsafe_allow_html=True)
        with st.form("add_asset_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                nuevo_ticker = st.text_input("Ticker")
            with c2:
                weight_actual = st.number_input("Peso actual (%)", value=0.0, step=1.0, min_value=0.0, max_value=100.0,
                                                help="Si ya lo tienes en tu portafolio, indica qué % representa. Si no lo tienes, déjalo en 0%.")
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                btn = st.form_submit_button("Agregar")
            if btn and nuevo_ticker:
                t = nuevo_ticker.strip().upper()
                if not t.endswith(".MX"):
                    t += ".MX"
                st.session_state['cartera'].append({"Ticker": t, "Peso": weight_actual / 100})
                st.rerun()
    with tab_popular:
        st.markdown(f"<p class='explainer'>Selecciona acciones de las más operadas en la Bolsa Mexicana.</p>", unsafe_allow_html=True)
        selected = st.multiselect("Selecciona activos", options=list(POPULAR_MX_TICKERS.keys()),
                                  format_func=lambda x: f"{x.replace('.MX','')} — {POPULAR_MX_TICKERS[x]}")
        if st.button("Agregar seleccionados") and selected:
            for t in selected:
                if not any(item['Ticker'] == t for item in st.session_state['cartera']):
                    st.session_state['cartera'].append({"Ticker": t, "Peso": 0.0})
            st.rerun()


def render_portfolio_pills():
    if not st.session_state['cartera']:
        return
    pills = ""
    for item in st.session_state['cartera']:
        name = POPULAR_MX_TICKERS.get(item['Ticker'], item['Ticker'].replace('.MX', ''))
        peso = f" ({item['Peso']:.0%})" if item['Peso'] > 0 else ""
        pills += f"<span class='asset-pill'>{name}{peso}</span>"
    st.markdown(pills, unsafe_allow_html=True)
    if len(st.session_state['cartera']) < 5:
        faltan = 5 - len(st.session_state['cartera'])
        st.markdown(f"<div class='alert-info'><b>Necesitas al menos 5 activos.</b> Te faltan <b>{faltan}</b> más.</div>", unsafe_allow_html=True)
    if st.button("Limpiar portafolio"):
        st.session_state['cartera'] = []
        st.rerun()


# ==========================================
# PAGE: MI PORTAFOLIO
# ==========================================
if menu == "Mi Portafolio":
    st.markdown(f"<h1 style='color:{TEAL_DARK};'>Mi Portafolio</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='explainer'>Arma tu portafolio, consulta el estado de tus activos y recibe recomendaciones inteligentes de compra/venta.</p>", unsafe_allow_html=True)

    st.markdown("<h3>1. Arma tu portafolio</h3>", unsafe_allow_html=True)
    render_add_assets()

    if st.session_state['cartera']:
        st.markdown(f"<h3>Tu portafolio ({len(st.session_state['cartera'])} activos)</h3>", unsafe_allow_html=True)
        render_portfolio_pills()

        if st.session_state['meta_model'] is not None:
            st.markdown("<span class='badge badge-ok'>Modelo IA entrenado</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warn'>Ejecuta la Simulación Histórica para entrenar el modelo IA</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Asset detail table
    if st.session_state['cartera']:
        st.markdown("<h3>2. Estado de tus activos</h3>", unsafe_allow_html=True)
        activos_lista = list(set([item["Ticker"] for item in st.session_state['cartera']]))
        with st.spinner("Consultando mercado..."):
            precios_preview, _ = fetch_market_data_cached(tuple(sorted(activos_lista)), "2y")
        if precios_preview is not None:
            summaries = []
            for t in activos_lista:
                s = get_asset_summary(precios_preview, t)
                if s:
                    s["Activo"] = POPULAR_MX_TICKERS.get(t, t.replace('.MX', ''))
                    s["Ticker"] = t
                    summaries.append(s)
            if summaries:
                df_s = pd.DataFrame(summaries)[["Activo", "Ticker", "Precio", "Rend. 1 Mes", "Rend. 3 Meses", "Rend. 1 Año", "Volatilidad Anual", "Dist. desde Máximo"]]
                st.dataframe(df_s.set_index("Activo"), use_container_width=True)
                st.markdown(f"""<div class='info-card'><h4>¿Cómo leer esta tabla?</h4><p>
                    <b>Rendimiento</b>: cuánto subió (+) o bajó (-) el precio en ese período.<br>
                    <b>Volatilidad</b>: qué tan bruscos son los movimientos. Mayor = más riesgo.<br>
                    <b>Dist. desde Máximo</b>: qué tan lejos está del precio más alto histórico.</p></div>""", unsafe_allow_html=True)

            # Price chart
            chart_tickers = [t for t in activos_lista if t in precios_preview.columns and t != "^MXX"]
            if chart_tickers and len(precios_preview) >= 252:
                normalized = precios_preview[chart_tickers].iloc[-252:].div(precios_preview[chart_tickers].iloc[-252].replace(0, np.nan)) * 100
                fig = go.Figure()
                colors = [TEAL_MID, TEAL_LIGHT, GRAY_500, WARN_SOFT, SUCCESS, "#9B8EC5", "#E8927C"]
                for idx, col in enumerate(normalized.columns):
                    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col],
                                             name=POPULAR_MX_TICKERS.get(col, col.replace('.MX', '')),
                                             line=dict(color=colors[idx % len(colors)], width=2)))
                confidelis_layout(fig, "Evolución último año (base 100)")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"<p class='explainer'>Todos parten de 100. Si una línea está en 120, subió 20%. Si está en 85, bajó 15%.</p>", unsafe_allow_html=True)

    # Orders
    if len(st.session_state['cartera']) > 4:
        st.markdown("---")
        st.markdown("<h3>3. Recomendaciones de inversión</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='explainer'>El sistema analiza tendencias, volatilidad y señales usando IA para sugerirte qué hacer mañana.</p>", unsafe_allow_html=True)

        df_cart = pd.DataFrame(st.session_state['cartera']).groupby("Ticker").sum().reset_index()
        activos_brutos = df_cart["Ticker"].tolist()
        cw_dict = dict(zip(df_cart["Ticker"], df_cart["Peso"]))
        capital = st.number_input("Capital invertido ($ MXN)", value=1000000, step=100000, min_value=100000,
                                  help="Monto total que tienes disponible para invertir.")

        if st.button("Generar recomendaciones para mañana"):
            with st.spinner("Analizando mercado con IA..."):
                precios, vol = fetch_market_data(activos_brutos, "5y")
                if precios is None:
                    st.error("No se pudo conectar. Verifica tu internet.")
                    st.stop()
                valid = [c for c in precios.columns if precios[c].count() >= 252 or c == "^MXX"]
                activos = [a for a in activos_brutos if a in valid]
                if len(activos) < 5:
                    st.error("No hay suficientes activos con historial (mínimo 5).")
                    st.stop()

                cw = np.array([cw_dict.get(a, 0) for a in activos])
                returns = calcular_retornos_robustos(precios)
                tp = precios.iloc[-504:] if len(precios) > 504 else precios
                tr = returns.iloc[-504:] if len(returns) > 504 else returns
                tv = vol.iloc[-504:] if len(vol) > 504 else vol

                a_stat = generate_multi_factor_alpha(tp[activos]).fillna(0)
                a_ml = generate_panel_ml_alpha(tp[activos], tr[activos], tv[activos]).fillna(0)
                regime = detect_regime_gmm(tr["^MXX"])

                if st.session_state['meta_model'] is not None:
                    cx = np.column_stack([[a_stat.get(a, 0) for a in activos], [a_ml.get(a, 0) for a in activos]])
                    meta_alpha = pd.Series(st.session_state['meta_model'].predict(cx), index=activos)
                else:
                    w_ml = 0.2 if regime == "Volátil" else 0.7
                    meta_alpha = ((1 - w_ml) * a_stat.reindex(activos).fillna(0)) + (w_ml * a_ml.reindex(activos).fillna(0))

                vf = tr[activos].ewm(span=30).std().iloc[-1] * np.sqrt(252)
                exp_ret = meta_alpha.values * vf.values
                cov = get_ledoit_wolf_cov(tr[activos])
                betas = calculate_betas(tr[activos], tr["^MXX"])
                adv = (tp[activos].tail(20) * tv[activos].tail(20)).mean().values
                adv_max = np.clip((0.10 * adv) / capital, 0.01, 0.30)
                target_vol = 0.08 if regime == "Volátil" else 0.15

                raw = optimizar_market_neutral_pro(exp_ret, cov, cw, betas, adv_max, vf.values, regime)
                cvol = np.sqrt(np.dot(raw.T, np.dot(cov, raw)))
                nw = raw * (target_vol / cvol) if cvol > 0 else raw

                if regime == "Volátil":
                    st.markdown(f"<div class='alert-crisis'><b>Mercado en fase volátil.</b> Se reduce el riesgo automáticamente para proteger tu capital.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-stable'><b>Mercado estable.</b> Condiciones favorables para operar con normalidad.</div>", unsafe_allow_html=True)

                bn = np.dot(nw, betas)
                c1, c2, c3 = st.columns(3)
                c1.metric("Riesgo esperado", f"{cvol:.1%}", help="Volatilidad anualizada del portafolio.")
                c2.metric("Beta (sensibilidad al mercado)", f"{bn:.3f}", help="Cercano a 0 = independiente del mercado.")
                c3.metric("Régimen", regime)

                df_o = pd.DataFrame({"Ticker": activos, "Activo": [POPULAR_MX_TICKERS.get(a, a.replace('.MX', '')) for a in activos],
                                     "Peso Actual": cw, "Peso Sugerido": nw})
                df_o["Cambio"] = df_o["Peso Sugerido"] - df_o["Peso Actual"]
                df_o["Monto ($)"] = df_o["Cambio"] * capital
                df_o["Acción"] = df_o["Cambio"].apply(lambda d: "Comprar" if d > 0.01 else ("Reducir" if d < -0.01 else "Mantener"))

                df_d = df_o.copy()
                for c in ["Peso Actual", "Peso Sugerido", "Cambio"]:
                    df_d[c] = df_d[c].apply(lambda x: f"{x:.1%}")
                df_d["Monto ($)"] = df_d["Monto ($)"].apply(lambda x: f"${x:+,.0f}")
                st.dataframe(df_d[["Activo", "Ticker", "Peso Actual", "Peso Sugerido", "Cambio", "Monto ($)", "Acción"]].set_index("Activo"), use_container_width=True)

                fig_w = go.Figure()
                names = [POPULAR_MX_TICKERS.get(a, a.replace('.MX', '')) for a in activos]
                fig_w.add_trace(go.Bar(name="Actual", x=names, y=df_o["Peso Actual"].values, marker_color=GRAY_400))
                fig_w.add_trace(go.Bar(name="Sugerido", x=names, y=df_o["Peso Sugerido"].values, marker_color=TEAL_MID))
                confidelis_layout(fig_w, "Actual vs. Recomendado")
                fig_w.update_layout(barmode='group')
                fig_w.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig_w, use_container_width=True)

                csv = df_d.to_csv(index=True).encode('utf-8')
                st.download_button("Descargar recomendaciones (CSV)", data=csv,
                                   file_name=f'confidelis_{datetime.now().strftime("%Y%m%d")}.csv', mime='text/csv')


# ==========================================
# PAGE: ANÁLISIS DE INVERSIÓN
# ==========================================
elif menu == "Análisis de Inversión":
    st.markdown(f"<h1 style='color:{TEAL_DARK};'>Análisis de Inversión</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div class='info-card'><h4>¿Cuánto he ganado o perdido?</h4><p>
        Ingresa cuánto invertiste en cada acción y la fecha aproximada. El sistema calcula cuánto vale hoy
        y te muestra tu ganancia o pérdida real.</p></div>""", unsafe_allow_html=True)

    st.markdown("<h3>Agrega tus inversiones</h3>", unsafe_allow_html=True)

    if 'inversiones' not in st.session_state:
        st.session_state['inversiones'] = []

    with st.form("inv_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            inv_ticker = st.selectbox("Activo", options=list(POPULAR_MX_TICKERS.keys()),
                                      format_func=lambda x: f"{x.replace('.MX','')} — {POPULAR_MX_TICKERS[x]}")
        with c2:
            inv_monto = st.number_input("Monto invertido ($)", value=50000, step=5000, min_value=1000)
        with c3:
            inv_fecha = st.date_input("Fecha de compra", value=datetime(2025, 1, 2))
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            inv_btn = st.form_submit_button("Agregar")
        if inv_btn:
            st.session_state['inversiones'].append({"Ticker": inv_ticker, "Monto": inv_monto, "Fecha": str(inv_fecha)})
            st.rerun()

    if st.session_state['inversiones']:
        df_inv = pd.DataFrame(st.session_state['inversiones'])
        st.markdown(f"<h4>Tus inversiones registradas</h4>", unsafe_allow_html=True)

        # Show registered investments
        df_show = df_inv.copy()
        df_show["Activo"] = df_show["Ticker"].map(lambda t: POPULAR_MX_TICKERS.get(t, t))
        df_show["Monto"] = df_show["Monto"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_show[["Activo", "Ticker", "Monto", "Fecha"]], use_container_width=True)

        if st.button("Limpiar inversiones"):
            st.session_state['inversiones'] = []
            st.rerun()

        if st.button("Calcular resultados"):
            with st.spinner("Calculando..."):
                all_tickers = list(df_inv["Ticker"].unique())
                precios, _ = fetch_market_data(all_tickers, period="5y")
                if precios is None:
                    st.error("No se pudieron obtener datos.")
                    st.stop()

                resultados = []
                for _, row in df_inv.iterrows():
                    t = row["Ticker"]
                    if t not in precios.columns:
                        continue
                    fecha_compra = pd.Timestamp(row["Fecha"])
                    p = precios[t].dropna()
                    # Encontrar precio más cercano a la fecha de compra
                    p_after = p[p.index >= fecha_compra]
                    if p_after.empty:
                        continue
                    precio_compra = p_after.iloc[0]
                    precio_actual = p.iloc[-1]
                    acciones = row["Monto"] / precio_compra
                    valor_actual = acciones * precio_actual
                    ganancia = valor_actual - row["Monto"]
                    rend = (precio_actual / precio_compra) - 1
                    dias = (p.index[-1] - p_after.index[0]).days
                    rend_anual = ((1 + rend) ** (365 / max(dias, 1))) - 1 if dias > 0 else 0

                    resultados.append({
                        "Activo": POPULAR_MX_TICKERS.get(t, t),
                        "Invertido": row["Monto"],
                        "Valor Actual": valor_actual,
                        "Ganancia/Pérdida": ganancia,
                        "Rendimiento": rend,
                        "Rend. Anualizado": rend_anual,
                        "Días": dias,
                    })

                if resultados:
                    df_r = pd.DataFrame(resultados)
                    total_inv = df_r["Invertido"].sum()
                    total_val = df_r["Valor Actual"].sum()
                    total_gan = total_val - total_inv
                    total_rend = total_val / total_inv - 1

                    st.markdown("---")
                    st.markdown("<h3>Resultados de tu inversión</h3>", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total invertido", f"${total_inv:,.0f}")
                    c2.metric("Valor actual", f"${total_val:,.0f}")
                    color_gan = SUCCESS if total_gan >= 0 else DANGER
                    c3.metric("Ganancia/Pérdida", f"${total_gan:+,.0f}")
                    c4.metric("Rendimiento total", f"{total_rend:+.1%}")

                    if total_gan >= 0:
                        st.markdown(f"<div class='alert-stable'><b>Tu portafolio ha generado valor.</b> Has ganado <b>${total_gan:,.0f}</b> pesos desde que invertiste.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='alert-crisis'><b>Tu portafolio tiene una pérdida.</b> La minusvalía actual es de <b>${abs(total_gan):,.0f}</b> pesos. Recuerda: las pérdidas no se materializan hasta que vendes.</div>", unsafe_allow_html=True)

                    # Detail table
                    df_d = df_r.copy()
                    df_d["Invertido"] = df_d["Invertido"].apply(lambda x: f"${x:,.0f}")
                    df_d["Valor Actual"] = df_d["Valor Actual"].apply(lambda x: f"${x:,.0f}")
                    df_d["Ganancia/Pérdida"] = df_d["Ganancia/Pérdida"].apply(lambda x: f"${x:+,.0f}")
                    df_d["Rendimiento"] = df_d["Rendimiento"].apply(lambda x: f"{x:+.1%}")
                    df_d["Rend. Anualizado"] = df_d["Rend. Anualizado"].apply(lambda x: f"{x:+.1%}")
                    st.dataframe(df_d.set_index("Activo"), use_container_width=True)

                    # Pie chart of current portfolio
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=df_r["Activo"], values=df_r["Valor Actual"],
                        hole=0.45,
                        marker=dict(colors=[TEAL_MID, TEAL_LIGHT, GRAY_500, WARN_SOFT, SUCCESS, "#9B8EC5", "#E8927C"]),
                        textinfo='label+percent',
                        textfont=dict(size=12)
                    )])
                    confidelis_layout(fig_pie, "Composición actual de tu portafolio")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Evolution chart
                    all_t = [row["Ticker"] for _, row in df_inv.iterrows()]
                    min_date = pd.Timestamp(df_inv["Fecha"].min())
                    prices_chart = precios[list(set(all_t))].loc[min_date:].dropna(how='all')
                    if not prices_chart.empty:
                        # Simulate portfolio value over time
                        port_value = pd.Series(0.0, index=prices_chart.index)
                        for _, row in df_inv.iterrows():
                            t = row["Ticker"]
                            if t not in prices_chart.columns:
                                continue
                            fecha = pd.Timestamp(row["Fecha"])
                            p_after = prices_chart[t][prices_chart.index >= fecha].dropna()
                            if p_after.empty:
                                continue
                            acciones = row["Monto"] / p_after.iloc[0]
                            contrib = prices_chart[t].ffill() * acciones
                            contrib = contrib[contrib.index >= fecha].reindex(port_value.index).fillna(0)
                            port_value = port_value + contrib

                        port_value = port_value[port_value > 0]
                        if len(port_value) > 10:
                            fig_ev = go.Figure()
                            fig_ev.add_trace(go.Scatter(x=port_value.index, y=port_value,
                                                        name="Valor de tu portafolio",
                                                        line=dict(color=TEAL_MID, width=2.5),
                                                        fill='tozeroy', fillcolor=f"rgba(61,140,140,0.08)"))
                            fig_ev.add_hline(y=total_inv, line_dash="dash", line_color=GRAY_400,
                                            annotation_text=f"Invertido: ${total_inv:,.0f}")
                            confidelis_layout(fig_ev, "Evolución del valor de tu portafolio")
                            fig_ev.update_yaxes(tickprefix="$", tickformat=",")
                            st.plotly_chart(fig_ev, use_container_width=True)
                            st.markdown(f"<p class='explainer'>La línea muestra cómo ha crecido (o disminuido) el valor de tu dinero desde que invertiste. La línea punteada gris es el monto que pusiste originalmente.</p>", unsafe_allow_html=True)

                    # Compare vs CETES
                    if cetes_auto and total_inv > 0:
                        max_dias = df_r["Días"].max()
                        cetes_ganancia = total_inv * (cetes_auto / 100) * (max_dias / 365)
                        st.markdown(f"""<div class='info-card'><h4>Comparación con CETES</h4><p>
                            Si hubieras puesto los mismos <b>${total_inv:,.0f}</b> en CETES a {cetes_auto:.2f}% anual,
                            habrías ganado aproximadamente <b>${cetes_ganancia:,.0f}</b> en {max_dias} días.
                            {"Tu portafolio lo superó." if total_gan > cetes_ganancia else "CETES habría sido mejor opción en este período."}
                        </p></div>""", unsafe_allow_html=True)
    else:
        st.info("Agrega al menos una inversión para ver el análisis.")


# ==========================================
# PAGE: SIMULACIÓN HISTÓRICA
# ==========================================
elif menu == "Simulación Histórica":
    st.markdown(f"<h1 style='color:{TEAL_DARK};'>Simulación Histórica</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div class='info-card'><h4>¿Qué es esto?</h4><p>
        Probamos cómo habría funcionado nuestra estrategia de IA <b>en el pasado</b> con datos reales.
        Es una "máquina del tiempo financiera" para validar que el modelo funciona antes de usarlo con dinero real.<br><br>
        <b>Importante:</b> Resultados pasados <b>no garantizan</b> resultados futuros.</p></div>""", unsafe_allow_html=True)

    if len(st.session_state['cartera']) < 5:
        st.markdown("<h3>Primero arma tu portafolio</h3>", unsafe_allow_html=True)
        render_add_assets()
        render_portfolio_pills()
    else:
        activos_brutos = list(set([item["Ticker"] for item in st.session_state['cartera']]))
        pills = "".join([f"<span class='asset-pill'>{POPULAR_MX_TICKERS.get(t, t.replace('.MX',''))}</span>" for t in activos_brutos])
        st.markdown(f"<h4>Activos a simular:</h4>{pills}", unsafe_allow_html=True)
        st.markdown("")

        if st.button("Ejecutar simulación completa"):
            pb = st.progress(0, "Iniciando...")
            pb.progress(10, "Descargando datos históricos...")
            precios, vol = fetch_market_data_cached(tuple(sorted(activos_brutos)), "5y")
            if precios is None:
                st.error("No se pudieron obtener datos.")
                st.stop()

            pb.progress(20, "Validando activos...")
            valid = [c for c in precios.columns if precios[c].count() >= 252 or c == "^MXX"]
            activos = [a for a in activos_brutos if a in valid]
            excl = set(activos_brutos) - set(activos)
            if excl:
                st.warning(f"Excluidos: {', '.join(POPULAR_MX_TICKERS.get(t, t) for t in excl)}")
            if len(activos) < 5:
                st.error("No hay suficientes activos con al menos 1 año de historial.")
                st.stop()

            pb.progress(30, "Entrenando modelos de IA (puede tomar un momento)...")
            oos_ret, trained_model, bench_ret = backtest_walk_forward_meta_model(precios, vol, activos, rf_daily, initial_capital=10000000)

            pb.progress(90, "Generando gráficas...")
            if trained_model is not None:
                st.session_state['meta_model'] = trained_model

            if len(oos_ret) > 0:
                total_ret = np.prod(1 + oos_ret) - 1
                ann_ret = (1 + total_ret) ** (252 / len(oos_ret)) - 1
                vol_oos = oos_ret.std() * np.sqrt(252)
                dsr, psr = probabilistic_sharpe_ratio(oos_ret, rf_daily)
                sharpe = (ann_ret - rf_input / 100) / vol_oos if vol_oos > 0 else 0
                dd, max_dd = calcular_drawdown_avanzado(oos_ret)

                pb.progress(100, "Listo")

                st.markdown("---")
                st.markdown("<h3>Resultados</h3>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rendimiento anual", f"{ann_ret:.1%}", help="CAGR: rendimiento compuesto anualizado.")
                c2.metric("Riesgo (volatilidad)", f"{vol_oos:.1%}", help="Menor = más estable.")
                c3.metric("Sharpe Ratio", f"{sharpe:.2f}", help=">1 bueno, >2 excelente.")
                c4.metric("Peor caída", f"{max_dd:.1%}", help="Máxima pérdida antes de recuperarse.")

                c5, c6 = st.columns(2)
                c5.metric("Confianza (PSR)", f"{psr:.0%}", help=">95% = estadísticamente significativo.")
                c6.metric("Sharpe ajustado (DSR)", f"{dsr:.2f}")

                if sharpe > 1.0 and psr > 0.85:
                    st.markdown(f"<div class='alert-stable'><b>Resultado sólido.</b> El modelo ha sido entrenado y está listo en 'Mi Portafolio'.</div>", unsafe_allow_html=True)
                elif sharpe > 0.3:
                    st.markdown(f"<div class='alert-info'><b>Resultado moderado.</b> Considera diversificar más.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-crisis'><b>Resultado débil.</b> Prueba con otros activos.</div>", unsafe_allow_html=True)

                eq = (1 + oos_ret).cumprod()
                eq_b = (1 + bench_ret).cumprod()
                fig = go.Figure()
                for path in monte_carlo_shuffle_test(oos_ret, 30):
                    fig.add_trace(go.Scatter(x=eq.index[:len(path)], y=path, mode='lines',
                                             line=dict(color=GRAY_300, width=1), opacity=0.12, showlegend=False))
                fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Estrategia Confidelis", line=dict(color=TEAL_MID, width=3)))
                fig.add_trace(go.Scatter(x=eq_b.index, y=eq_b, name="IPC (Bolsa Mexicana)", line=dict(color=GRAY_500, width=2, dash='dot')))
                confidelis_layout(fig, "Crecimiento de $1 invertido")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"<p class='explainer'>Línea teal = estrategia IA. Línea gris punteada = simplemente invertir en la bolsa. Líneas grises claras = simulaciones aleatorias para validar que no es azar.</p>", unsafe_allow_html=True)

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name="Caída", line=dict(color=DANGER, width=2), fillcolor="rgba(176,64,80,0.1)"))
                confidelis_layout(fig_dd, "Caídas del portafolio (drawdown)")
                fig_dd.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                pb.progress(100, "Completado")
                st.warning("No hubo suficientes datos para la simulación. Usa activos con más historial.")

