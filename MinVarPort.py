import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="ESG Portfolio Optimiser", layout="wide")

# =========================================================
# Defaults
# =========================================================
DEFAULTS = {
    "page": "intro",
    "mu1_pct": 5.00,
    "mu2_pct": 12.00,
    "sigma1_pct": 9.00,
    "sigma2_pct": 20.00,
    "rf_pct": 2.00,
    "rho": -0.20,
    "esg1": 35.0,
    "esg2": 80.0,
    "lambda_esg": 0.30,
    "gamma": 3.0,
    "num_points": 1001,
    "firm_frontier_points": 80,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# =========================================================
# Finance / ESG functions (2-asset teaching model)
# =========================================================
def var_covar(sigmas: np.ndarray, rho: float) -> np.ndarray:
    """2x2 covariance matrix."""
    return np.array([
        [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
        [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2]
    ])


def build_portfolio_grid(
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: float,
    rf: float,
    esg_scores: np.ndarray,
    gamma: float,
    lambda_esg: float,
    num_points: int
) -> pd.DataFrame:
    """
    Build long-only portfolio grid for 2 risky assets.
    Inputs use decimals internally:
      returns: 0.05 = 5%
      volatilities: 0.09 = 9%
      ESG scores: 0.35 = 35/100
    """
    cov = var_covar(sigma, rho)
    weights = np.linspace(0, 1, num_points)

    rows = []
    for w1 in weights:
        w = np.array([w1, 1 - w1])

        exp_return = float(np.dot(mu, w))
        variance = float(np.dot(w, np.dot(cov, w)))
        std_dev = float(np.sqrt(max(variance, 0.0)))
        esg_score = float(np.dot(esg_scores, w))

        sharpe = np.nan if std_dev == 0 else (exp_return - rf) / std_dev

        # Utility:
        # U = E[R_p] - (gamma/2) * sigma_p^2 + lambda * s_bar
        utility = exp_return - 0.5 * gamma * variance + lambda_esg * esg_score

        rows.append({
            "Weight Asset 1": w1,
            "Weight Asset 2": 1 - w1,
            "Expected Return": exp_return,
            "Variance": variance,
            "Std Dev": std_dev,
            "ESG Score": esg_score,
            "Sharpe Ratio": sharpe,
            "Utility": utility,
        })

    return pd.DataFrame(rows)


def required_esg_threshold(df: pd.DataFrame, lambda_esg: float) -> float:
    """
    Convert lambda into a stricter ESG screen:
    required ESG = min ESG + lambda * (max ESG - min ESG)
    """
    s_min = float(df["ESG Score"].min())
    s_max = float(df["ESG Score"].max())
    return s_min + lambda_esg * (s_max - s_min)


def select_key_portfolios(df: pd.DataFrame):
    """Return minimum-variance and tangency portfolios."""
    valid = df[np.isfinite(df["Sharpe Ratio"])].copy()

    idx_mvp = df["Std Dev"].idxmin()
    idx_tan = valid["Sharpe Ratio"].idxmax()

    mvp = df.loc[idx_mvp]
    tangency = df.loc[idx_tan]

    return mvp, tangency


def summary_df(mvp: pd.Series, tangency: pd.Series, labels: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Portfolio": labels[0],
            "Weight Asset 1": mvp["Weight Asset 1"],
            "Weight Asset 2": mvp["Weight Asset 2"],
            "Expected Return": mvp["Expected Return"],
            "Std Dev": mvp["Std Dev"],
            "ESG Score": mvp["ESG Score"],
            "Sharpe Ratio": mvp["Sharpe Ratio"],
            "Utility": mvp["Utility"],
        },
        {
            "Portfolio": labels[1],
            "Weight Asset 1": tangency["Weight Asset 1"],
            "Weight Asset 2": tangency["Weight Asset 2"],
            "Expected Return": tangency["Expected Return"],
            "Std Dev": tangency["Std Dev"],
            "ESG Score": tangency["ESG Score"],
            "Sharpe Ratio": tangency["Sharpe Ratio"],
            "Utility": tangency["Utility"],
        },
    ])


def format_table(df: pd.DataFrame):
    return df.style.format({
        "Weight Asset 1": "{:.2%}",
        "Weight Asset 2": "{:.2%}",
        "Expected Return": "{:.2%}",
        "Std Dev": "{:.2%}",
        "ESG Score": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Utility": "{:.4f}",
    })


# =========================================================
# Uploaded firm-data loaders
# =========================================================
@st.cache_data
def load_firm_universe_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the uploaded 2024 firm universe and the correlation/covariance matrices.

    Files expected:
      - CRSP_merged_overlap_only.xlsx
      - CRSP_correlation_covariance_2010_2024.xlsx

    For consistency with 2024 annual returns, the firm-level optimiser builds an
    annual covariance matrix as:

        annual_cov = diag(annualised vol 2024) @ correlation_matrix @ diag(annualised vol 2024)

    Only firms with all required information are retained.
    """
    merged_candidates = [
        Path("CRSP_merged_overlap_only.xlsx"),
        Path("/mnt/CRSP_merged_overlap_only.xlsx"),
    ]
    matrix_candidates = [
        Path("CRSP_correlation_covariance_2010_2024.xlsx"),
        Path("/mnt/CRSP_correlation_covariance_2010_2024.xlsx"),
    ]

    merged_path = next((p for p in merged_candidates if p.exists()), None)
    matrix_path = next((p for p in matrix_candidates if p.exists()), None)

    if merged_path is None:
        raise FileNotFoundError(
            "Could not find 'CRSP_merged_overlap_only.xlsx'. Put it in the same folder as app.py."
        )
    if matrix_path is None:
        raise FileNotFoundError(
            "Could not find 'CRSP_correlation_covariance_2010_2024.xlsx'. Put it in the same folder as app.py."
        )

    merged = pd.read_excel(merged_path, sheet_name="Merged Data")
    corr = pd.read_excel(matrix_path, sheet_name="Correlation", index_col=0)
    raw_cov = pd.read_excel(matrix_path, sheet_name="Covariance", index_col=0)

    required_cols = {
        "ticker",
        "comname",
        "valuescore",
        "value",
        "return_2024",
        "annualized_volatility_std_dev",
    }
    missing = required_cols - set(merged.columns)
    if missing:
        raise ValueError(f"Merged file is missing required columns: {sorted(missing)}")

    # Keep only tickers that appear in all data sources
    overlap = sorted(
        set(merged["ticker"].astype(str))
        & set(corr.index.astype(str))
        & set(raw_cov.index.astype(str))
    )
    use = merged[merged["ticker"].astype(str).isin(overlap)].copy()

    # If any ticker appears multiple times, collapse to one row per ticker.
    # Numeric columns are averaged; text columns keep the first available value.
    agg_map = {}
    for col in use.columns:
        if col == "ticker":
            continue
        if pd.api.types.is_numeric_dtype(use[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"

    firms = use.groupby("ticker", as_index=False).agg(agg_map).copy()

    firms["valuescore"] = pd.to_numeric(firms["valuescore"], errors="coerce")
    firms["return_2024"] = pd.to_numeric(firms["return_2024"], errors="coerce")
    firms["annualized_volatility_std_dev"] = pd.to_numeric(
        firms["annualized_volatility_std_dev"], errors="coerce"
    )

    firms = firms.dropna(subset=["ticker", "valuescore", "return_2024", "annualized_volatility_std_dev"]).copy()
    firms = firms[firms["annualized_volatility_std_dev"] > 0].copy()

    tickers = firms["ticker"].astype(str).tolist()
    corr_sub = corr.loc[tickers, tickers].apply(pd.to_numeric, errors="coerce")
    cov_sub = raw_cov.loc[tickers, tickers].apply(pd.to_numeric, errors="coerce")

    # Drop any tickers with incomplete matrix rows/columns
    bad_tickers = set(corr_sub.index[corr_sub.isna().any(axis=1)])
    bad_tickers |= set(corr_sub.columns[corr_sub.isna().any(axis=0)])
    bad_tickers |= set(cov_sub.index[cov_sub.isna().any(axis=1)])
    bad_tickers |= set(cov_sub.columns[cov_sub.isna().any(axis=0)])

    firms = firms[~firms["ticker"].isin(bad_tickers)].copy().sort_values("ticker").reset_index(drop=True)
    tickers = firms["ticker"].astype(str).tolist()
    corr_sub = corr.loc[tickers, tickers].apply(pd.to_numeric, errors="coerce")

    # Build an annual covariance matrix consistent with 2024 annual returns
    vols = firms["annualized_volatility_std_dev"].to_numpy(dtype=float)
    annual_cov = np.outer(vols, vols) * corr_sub.to_numpy(dtype=float)

    # Numerical clean-up
    annual_cov = (annual_cov + annual_cov.T) / 2.0
    np.fill_diagonal(annual_cov, np.maximum(np.diag(annual_cov), 1e-10))

    # Small ridge to stabilise inversion in large universes
    annual_cov = annual_cov + np.eye(len(firms)) * 1e-8

    rename_map = {
        "ticker": "Ticker",
        "comname": "Company",
        "value": "ESG Grade",
        "valuescore": "ESG Score",
        "return_2024": "Expected Return",
        "annualized_volatility_std_dev": "Annual Volatility",
    }
    firms = firms.rename(columns=rename_map)

    if "isin" not in firms.columns:
        firms["isin"] = ""
    firms = firms.rename(columns={"isin": "ISIN"})

    firms["ESG Score (%)"] = firms["ESG Score"] * 100
    firms["Expected Return (%)"] = firms["Expected Return"] * 100
    firms["Annual Volatility (%)"] = firms["Annual Volatility"] * 100

    annual_cov_df = pd.DataFrame(annual_cov, index=firms["Ticker"], columns=firms["Ticker"])
    corr_sub.index = firms["Ticker"]
    corr_sub.columns = firms["Ticker"]

    return firms.reset_index(drop=True), corr_sub, annual_cov_df


# =========================================================
# Multi-stock Markowitz helpers (firm universe)
# =========================================================
def invert_covariance(cov: np.ndarray) -> np.ndarray:
    """Stable inverse for the firm-level covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    return np.linalg.pinv(cov)


def frontier_constants(mu: np.ndarray, cov: np.ndarray):
    mu = np.asarray(mu, dtype=float)
    inv_cov = invert_covariance(cov)
    ones = np.ones(len(mu))

    A = float(ones @ inv_cov @ ones)
    B = float(ones @ inv_cov @ mu)
    C = float(mu @ inv_cov @ mu)
    D = max(A * C - B**2, 1e-12)
    return inv_cov, ones, A, B, C, D


def gmv_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv_cov, ones, A, _, _, _ = frontier_constants(mu, cov)
    w = (inv_cov @ ones) / A
    return w


def target_return_weights(mu: np.ndarray, cov: np.ndarray, target_return: float) -> np.ndarray:
    inv_cov, ones, A, B, C, D = frontier_constants(mu, cov)
    alpha = (C - B * target_return) / D
    beta = (A * target_return - B) / D
    w = inv_cov @ (alpha * ones + beta * mu)
    return w


def tangency_weights(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    inv_cov = invert_covariance(cov)
    ones = np.ones(len(mu))
    excess = mu - rf * ones
    raw = inv_cov @ excess
    denom = float(ones @ raw)
    if abs(denom) < 1e-12:
        return gmv_weights(mu, cov)
    return raw / denom


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, esg: np.ndarray | None = None) -> dict:
    weights = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    exp_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    std_dev = float(np.sqrt(max(variance, 0.0)))
    sharpe = np.nan if std_dev <= 1e-12 else exp_return / std_dev

    out = {
        "Expected Return": exp_return,
        "Variance": variance,
        "Std Dev": std_dev,
        "Sharpe Ratio (rf=0)": sharpe,
    }
    if esg is not None:
        out["ESG Score"] = float(weights @ esg)
    return out


def build_analytic_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    esg: np.ndarray,
    rf: float,
    num_points: int = 80,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build the classic unconstrained Markowitz frontier.
    This is used for the firm universe because it is fast and scales well.
    """
    mu = np.asarray(mu, dtype=float)
    esg = np.asarray(esg, dtype=float)
    cov = np.asarray(cov, dtype=float)

    w_gmv = gmv_weights(mu, cov)
    gmv = portfolio_stats(w_gmv, mu, cov, esg)
    gmv["Weights"] = w_gmv

    w_tan = tangency_weights(mu, cov, rf)
    tan = portfolio_stats(w_tan, mu - rf, cov, esg)
    tan["Expected Return"] = float(w_tan @ mu)
    tan["Std Dev"] = float(np.sqrt(max(float(w_tan @ cov @ w_tan), 0.0)))
    tan["Sharpe Ratio"] = np.nan if tan["Std Dev"] <= 1e-12 else (tan["Expected Return"] - rf) / tan["Std Dev"]
    tan["Weights"] = w_tan

    target_min = float(min(mu.min(), gmv["Expected Return"]))
    target_max = float(max(mu.max(), tan["Expected Return"], gmv["Expected Return"]))
    if np.isclose(target_min, target_max):
        target_max = target_min + 1e-6

    target_returns = np.linspace(target_min, target_max, num_points)

    rows = []
    for target in target_returns:
        w = target_return_weights(mu, cov, target)
        stats = portfolio_stats(w, mu, cov, esg)
        rows.append({
            "Target Return": target,
            "Expected Return": stats["Expected Return"],
            "Std Dev": stats["Std Dev"],
            "Variance": stats["Variance"],
            "ESG Score": stats["ESG Score"],
        })

    frontier_df = pd.DataFrame(rows)
    frontier_df = frontier_df.sort_values("Std Dev").reset_index(drop=True)
    frontier_df["Efficient"] = frontier_df["Expected Return"] >= gmv["Expected Return"] - 1e-12

    return frontier_df, gmv, tan


def weights_to_table(weights: np.ndarray, firms: pd.DataFrame, label: str) -> pd.DataFrame:
    out = firms[["Ticker", "Company", "ISIN", "ESG Grade", "ESG Score", "Expected Return", "Annual Volatility"]].copy()
    out["Weight"] = weights
    out["Abs Weight"] = out["Weight"].abs()
    out["Portfolio"] = label
    out = out.sort_values("Abs Weight", ascending=False).reset_index(drop=True)
    return out


def firm_screen_threshold(asset1_esg_pct: float, asset2_esg_pct: float, portfolio_esg_cutoff: float) -> float:
    """
    Default stock-level ESG cutoff used for the real-stock universe.
    We use the stricter of:
      - the lower of the two input asset ESG scores
      - the portfolio ESG cutoff implied by lambda
    """
    asset_floor = min(asset1_esg_pct, asset2_esg_pct) / 100.0
    return max(asset_floor, portfolio_esg_cutoff)


def format_weight_table(df: pd.DataFrame):
    return df.style.format({
        "ESG Score": "{:.2%}",
        "Expected Return": "{:.2%}",
        "Annual Volatility": "{:.2%}",
        "Weight": "{:.2%}",
        "Abs Weight": "{:.2%}",
    })


# =========================================================
# Page 1: Introduction
# =========================================================
if st.session_state.page == "intro":
    st.title("ESG Portfolio Optimiser")

    st.markdown(
        r"""
        This app compares:

        - a **standard mean-variance setup** using **all portfolios**
        - an **ESG-screened setup** using only portfolios that satisfy a **minimum portfolio ESG score**
        - a **firm-level optimiser** using the uploaded 2024 stock dataset plus the uploaded correlation matrix

        The investor utility in the 2-asset model is:

        \[
        U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda \bar{s}
        \]

        where:

        - \(E[R_p]\): expected portfolio return  
        - \(\sigma_p\): portfolio standard deviation  
        - \(\gamma\): risk aversion  
        - \(\bar{s}\): weighted average portfolio ESG score  
        - \(\lambda\): ESG preference intensity  

        In the firm-level tab, the app uses the uploaded real-stock universe and computes:

        - a frontier using **all firms with complete data**
        - a frontier using only **firms that pass the ESG screen**
        - the **minimum-variance portfolio** for the ESG-screened firm universe
        """
    )

    if st.button("Continue"):
        go_to("inputs")


# =========================================================
# Page 2: Inputs
# =========================================================
elif st.session_state.page == "inputs":
    st.title("Portfolio Inputs")

    st.write("Enter all percentages as values from 0 to 100.")

    with st.form("input_form"):
        st.subheader("Asset 1 inputs")
        mu1_pct = st.number_input(
            "Expected return for Asset 1 (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.mu1_pct),
            step=0.25,
            format="%.2f",
        )
        sigma1_pct = st.number_input(
            "Volatility for Asset 1 (%)",
            min_value=0.01,
            max_value=100.0,
            value=float(st.session_state.sigma1_pct),
            step=0.25,
            format="%.2f",
        )
        esg1 = st.number_input(
            "ESG score for Asset 1 (0 to 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.esg1),
            step=1.0,
            format="%.1f",
        )

        st.markdown("---")

        st.subheader("Asset 2 inputs")
        mu2_pct = st.number_input(
            "Expected return for Asset 2 (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.mu2_pct),
            step=0.25,
            format="%.2f",
        )
        sigma2_pct = st.number_input(
            "Volatility for Asset 2 (%)",
            min_value=0.01,
            max_value=100.0,
            value=float(st.session_state.sigma2_pct),
            step=0.25,
            format="%.2f",
        )
        esg2 = st.number_input(
            "ESG score for Asset 2 (0 to 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.esg2),
            step=1.0,
            format="%.1f",
        )

        st.markdown("---")

        st.subheader("Portfolio inputs")
        rf_pct = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.rf_pct),
            step=0.25,
            format="%.2f",
        )
        rho = st.slider(
            "Correlation between Asset 1 and Asset 2",
            min_value=-1.0,
            max_value=1.0,
            value=float(st.session_state.rho),
            step=0.01,
        )
        num_points = st.slider(
            "Number of portfolio weight points",
            min_value=101,
            max_value=5001,
            value=int(st.session_state.num_points),
            step=100,
        )

        st.markdown("---")

        st.subheader("Investor preferences")
        lambda_esg = st.slider(
            "ESG preference intensity λ",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.lambda_esg),
            step=0.01,
        )
        gamma = st.number_input(
            "Risk aversion γ",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.gamma),
            step=0.10,
            format="%.2f",
        )

        firm_frontier_points = st.slider(
            "Number of points for firm-level frontier",
            min_value=30,
            max_value=200,
            value=int(st.session_state.firm_frontier_points),
            step=10,
        )

        submitted = st.form_submit_button("Continue to results")

    if submitted:
        st.session_state.mu1_pct = mu1_pct
        st.session_state.mu2_pct = mu2_pct
        st.session_state.sigma1_pct = sigma1_pct
        st.session_state.sigma2_pct = sigma2_pct
        st.session_state.esg1 = esg1
        st.session_state.esg2 = esg2
        st.session_state.rf_pct = rf_pct
        st.session_state.rho = rho
        st.session_state.num_points = num_points
        st.session_state.lambda_esg = lambda_esg
        st.session_state.gamma = gamma
        st.session_state.firm_frontier_points = firm_frontier_points
        go_to("results")

    if st.button("Back to introduction"):
        go_to("intro")


# =========================================================
# Page 3: Results
# =========================================================
elif st.session_state.page == "results":
    st.title("Results")

    # Convert user inputs from % to decimals
    mu = np.array([st.session_state.mu1_pct, st.session_state.mu2_pct]) / 100.0
    sigma = np.array([st.session_state.sigma1_pct, st.session_state.sigma2_pct]) / 100.0
    rf = st.session_state.rf_pct / 100.0
    rho = st.session_state.rho
    esg_scores = np.array([st.session_state.esg1, st.session_state.esg2]) / 100.0
    lambda_esg = st.session_state.lambda_esg
    gamma = st.session_state.gamma
    num_points = st.session_state.num_points

    # All portfolios
    df_all = build_portfolio_grid(
        mu=mu,
        sigma=sigma,
        rho=rho,
        rf=rf,
        esg_scores=esg_scores,
        gamma=gamma,
        lambda_esg=lambda_esg,
        num_points=num_points,
    )

    # ESG-screened portfolios
    esg_cutoff = required_esg_threshold(df_all, lambda_esg)
    df_esg = df_all[df_all["ESG Score"] >= esg_cutoff - 1e-12].copy()

    # Safety fallback
    if df_esg.empty:
        idx_best_esg = df_all["ESG Score"].idxmax()
        df_esg = df_all.loc[[idx_best_esg]].copy()

    # Key portfolios
    mvp_std, tan_std = select_key_portfolios(df_all)
    mvp_esg, tan_esg = select_key_portfolios(df_esg)

    std_summary = summary_df(
        mvp_std,
        tan_std,
        labels=("Minimum Variance Portfolio", "Tangency Portfolio")
    )

    esg_summary = summary_df(
        mvp_esg,
        tan_esg,
        labels=("ESG Minimum Variance Portfolio", "ESG Tangency Portfolio")
    )

    st.markdown(
        f"""
        **ESG screen used in the 2-asset ESG graph**  
        Required portfolio ESG score = **{esg_cutoff * 100:.2f} / 100**

        This threshold is implied by your ESG preference \( \lambda = {lambda_esg:.2f} \).
        """
    )

    tab_analysis, tab_firm = st.tabs(["Portfolio analysis", "Firm-level frontier & ESG MVP"])

    with tab_analysis:
        # -----------------------------------------------------
        # Graph 1: Standard (all portfolios)
        # -----------------------------------------------------
        st.subheader("1) Standard mean-variance frontier and CML")

        fig1, ax1 = plt.subplots(figsize=(10, 6))

        x_all = df_all["Std Dev"] * 100
        y_all = df_all["Expected Return"] * 100
        rf_plot = rf * 100

        ax1.plot(
            x_all,
            y_all,
            color="blue",
            linewidth=2,
            label="Mean-variance frontier (all portfolios)"
        )

        # CML
        x_max_1 = max(float(x_all.max()), float(tan_std["Std Dev"] * 100)) * 1.10
        sigma_line_1 = np.linspace(0, x_max_1, 200)
        cml_1 = rf_plot + float(tan_std["Sharpe Ratio"]) * sigma_line_1

        ax1.plot(
            sigma_line_1,
            cml_1,
            color="blue",
            linestyle="--",
            linewidth=2,
            label="CML"
        )

        # Risk-free point
        ax1.scatter(
            [0],
            [rf_plot],
            color="black",
            s=70,
            label="Risk-free rate"
        )

        # MVP
        ax1.scatter(
            [mvp_std["Std Dev"] * 100],
            [mvp_std["Expected Return"] * 100],
            color="blue",
            marker="o",
            s=120,
            label="Minimum variance portfolio"
        )

        # Tangency
        ax1.scatter(
            [tan_std["Std Dev"] * 100],
            [tan_std["Expected Return"] * 100],
            color="blue",
            marker="*",
            s=220,
            label="Tangency portfolio"
        )

        ax1.annotate(
            "MVP",
            xy=(mvp_std["Std Dev"] * 100, mvp_std["Expected Return"] * 100),
            xytext=(8, 8),
            textcoords="offset points"
        )
        ax1.annotate(
            "Tangency",
            xy=(tan_std["Std Dev"] * 100, tan_std["Expected Return"] * 100),
            xytext=(8, -14),
            textcoords="offset points"
        )

        ax1.set_xlabel("Portfolio standard deviation (%)")
        ax1.set_ylabel("Expected return (%)")
        ax1.set_title("Standard frontier: all portfolios")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        ax1.grid(True)
        ax1.legend()

        st.pyplot(fig1)

        st.subheader("Summary table: Standard graph")
        st.dataframe(format_table(std_summary), use_container_width=True)

        # -----------------------------------------------------
        # Graph 2: ESG-screened only
        # -----------------------------------------------------
        st.subheader("2) ESG-screened frontier and CML")

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        x_esg = df_esg["Std Dev"] * 100
        y_esg = df_esg["Expected Return"] * 100

        ax2.plot(
            x_esg,
            y_esg,
            color="green",
            linewidth=2.5,
            label="ESG frontier (portfolios meeting minimum ESG score)"
        )

        # ESG CML
        x_max_2 = max(float(x_esg.max()), float(tan_esg["Std Dev"] * 100)) * 1.10
        sigma_line_2 = np.linspace(0, x_max_2, 200)
        cml_2 = rf_plot + float(tan_esg["Sharpe Ratio"]) * sigma_line_2

        ax2.plot(
            sigma_line_2,
            cml_2,
            color="green",
            linestyle="--",
            linewidth=2,
            label="ESG CML"
        )

        # Risk-free point
        ax2.scatter(
            [0],
            [rf_plot],
            color="black",
            s=70,
            label="Risk-free rate"
        )

        # ESG MVP
        ax2.scatter(
            [mvp_esg["Std Dev"] * 100],
            [mvp_esg["Expected Return"] * 100],
            color="green",
            marker="o",
            s=120,
            label="ESG minimum variance portfolio"
        )

        # ESG Tangency
        ax2.scatter(
            [tan_esg["Std Dev"] * 100],
            [tan_esg["Expected Return"] * 100],
            color="green",
            marker="*",
            s=220,
            label="ESG tangency portfolio"
        )

        ax2.annotate(
            "ESG MVP",
            xy=(mvp_esg["Std Dev"] * 100, mvp_esg["Expected Return"] * 100),
            xytext=(8, 8),
            textcoords="offset points"
        )
        ax2.annotate(
            "ESG Tangency",
            xy=(tan_esg["Std Dev"] * 100, tan_esg["Expected Return"] * 100),
            xytext=(8, -14),
            textcoords="offset points"
        )

        ax2.set_xlabel("Portfolio standard deviation (%)")
        ax2.set_ylabel("Expected return (%)")
        ax2.set_title("ESG-screened frontier: portfolios with minimum ESG score")
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig2)

        st.subheader("Summary table: ESG graph")
        st.dataframe(format_table(esg_summary), use_container_width=True)

        with st.expander("Show full portfolio table"):
            st.dataframe(
                df_all.style.format({
                    "Weight Asset 1": "{:.2%}",
                    "Weight Asset 2": "{:.2%}",
                    "Expected Return": "{:.2%}",
                    "Variance": "{:.5f}",
                    "Std Dev": "{:.2%}",
                    "ESG Score": "{:.2%}",
                    "Sharpe Ratio": "{:.3f}",
                    "Utility": "{:.4f}",
                }),
                use_container_width=True,
                height=350
            )

        csv_data = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full portfolio table as CSV",
            data=csv_data,
            file_name="esg_portfolio_results.csv",
            mime="text/csv"
        )

    with tab_firm:
        st.subheader("3) Real-stock frontier and ESG minimum-variance portfolio")

        st.markdown(
            """
            This tab uses the uploaded stock universe and calculates a **true firm-level** mean-variance setup.

            It compares:
            - a frontier built from **all firms with complete data**
            - a frontier built only from firms that pass the **minimum company ESG score**
            - the **minimum-variance portfolio** from the ESG-screened firm universe

            **Important note:** to keep units consistent with the 2024 annual returns,
            the optimiser combines the uploaded **correlation matrix** with the uploaded
            **2024 annualised volatilities** to build the covariance matrix used here.

            The firm-level frontier below is the classic **unconstrained Markowitz frontier**.
            """
        )

        try:
            firms, corr_sub, annual_cov_df = load_firm_universe_data()

            default_company_cutoff = firm_screen_threshold(
                asset1_esg_pct=st.session_state.esg1,
                asset2_esg_pct=st.session_state.esg2,
                portfolio_esg_cutoff=esg_cutoff,
            )

            st.write(f"Usable firms with complete information: **{len(firms)}**")
            st.write(
                f"Default company ESG cutoff (based on your asset ESG inputs and the portfolio ESG screen): "
                f"**{default_company_cutoff * 100:.2f} / 100**"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                min_company_esg_pct = st.number_input(
                    "Minimum company ESG score for firm frontier (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_company_cutoff * 100),
                    step=1.0,
                    format="%.1f",
                    key="min_company_esg_pct_firm_frontier"
                )
            with col_b:
                firm_frontier_points = st.slider(
                    "Points used for firm frontier graph",
                    min_value=30,
                    max_value=200,
                    value=int(st.session_state.firm_frontier_points),
                    step=10,
                    key="firm_frontier_points_results"
                )

            # All-stock universe
            firms_all = firms.copy()
            mu_all = firms_all["Expected Return"].to_numpy(dtype=float)
            esg_all = firms_all["ESG Score"].to_numpy(dtype=float)
            cov_all = annual_cov_df.loc[firms_all["Ticker"], firms_all["Ticker"]].to_numpy(dtype=float)

            frontier_all, gmv_all, tan_all = build_analytic_frontier(
                mu=mu_all,
                cov=cov_all,
                esg=esg_all,
                rf=rf,
                num_points=firm_frontier_points,
            )

            # ESG-screened firm universe
            cutoff = min_company_esg_pct / 100.0
            firms_esg = firms_all[firms_all["ESG Score"] >= cutoff].copy().reset_index(drop=True)

            st.write(f"Firms meeting the company ESG cutoff: **{len(firms_esg)}**")

            if len(firms_esg) < 2:
                st.warning("Fewer than 2 firms meet the company ESG cutoff. Lower the minimum company ESG score.")
            else:
                mu_esg_firms = firms_esg["Expected Return"].to_numpy(dtype=float)
                esg_esg_firms = firms_esg["ESG Score"].to_numpy(dtype=float)
                cov_esg = annual_cov_df.loc[firms_esg["Ticker"], firms_esg["Ticker"]].to_numpy(dtype=float)

                frontier_esg, gmv_esg_firms, tan_esg_firms = build_analytic_frontier(
                    mu=mu_esg_firms,
                    cov=cov_esg,
                    esg=esg_esg_firms,
                    rf=rf,
                    num_points=firm_frontier_points,
                )

                # -------------------------------------------------
                # Graph: two firm-level frontiers
                # -------------------------------------------------
                st.subheader("Firm-level mean-variance frontiers")
                fig3, ax3 = plt.subplots(figsize=(11, 6))

                all_eff = frontier_all[frontier_all["Efficient"]].copy()
                esg_eff = frontier_esg[frontier_esg["Efficient"]].copy()

                ax3.plot(
                    all_eff["Std Dev"] * 100,
                    all_eff["Expected Return"] * 100,
                    linewidth=2.2,
                    label="Frontier: all complete-data firms"
                )
                ax3.plot(
                    esg_eff["Std Dev"] * 100,
                    esg_eff["Expected Return"] * 100,
                    linewidth=2.2,
                    label="Frontier: ESG-screened firms only"
                )

                ax3.scatter(
                    [gmv_all["Std Dev"] * 100],
                    [gmv_all["Expected Return"] * 100],
                    marker="o",
                    s=100,
                    label="GMV: all firms"
                )
                ax3.scatter(
                    [gmv_esg_firms["Std Dev"] * 100],
                    [gmv_esg_firms["Expected Return"] * 100],
                    marker="*",
                    s=220,
                    label="GMV: ESG-screened firms"
                )

                ax3.annotate(
                    "All-firm GMV",
                    xy=(gmv_all["Std Dev"] * 100, gmv_all["Expected Return"] * 100),
                    xytext=(8, 8),
                    textcoords="offset points"
                )
                ax3.annotate(
                    "ESG GMV",
                    xy=(gmv_esg_firms["Std Dev"] * 100, gmv_esg_firms["Expected Return"] * 100),
                    xytext=(8, -14),
                    textcoords="offset points"
                )

                ax3.set_xlabel("Portfolio standard deviation (%)")
                ax3.set_ylabel("Expected return (%)")
                ax3.set_title("Firm-level frontiers: all firms vs ESG-screened firms")
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)

                # -------------------------------------------------
                # Summary tables
                # -------------------------------------------------
                st.subheader("Firm-level portfolio summary")
                firm_summary = pd.DataFrame([
                    {
                        "Portfolio": "All-firm GMV",
                        "Expected Return": gmv_all["Expected Return"],
                        "Std Dev": gmv_all["Std Dev"],
                        "ESG Score": gmv_all["ESG Score"],
                    },
                    {
                        "Portfolio": "ESG-screened GMV",
                        "Expected Return": gmv_esg_firms["Expected Return"],
                        "Std Dev": gmv_esg_firms["Std Dev"],
                        "ESG Score": gmv_esg_firms["ESG Score"],
                    },
                    {
                        "Portfolio": "All-firm Tangency",
                        "Expected Return": tan_all["Expected Return"],
                        "Std Dev": tan_all["Std Dev"],
                        "ESG Score": tan_all["ESG Score"],
                    },
                    {
                        "Portfolio": "ESG-screened Tangency",
                        "Expected Return": tan_esg_firms["Expected Return"],
                        "Std Dev": tan_esg_firms["Std Dev"],
                        "ESG Score": tan_esg_firms["ESG Score"],
                    },
                ])

                st.dataframe(
                    firm_summary.style.format({
                        "Expected Return": "{:.2%}",
                        "Std Dev": "{:.2%}",
                        "ESG Score": "{:.2%}",
                    }),
                    use_container_width=True,
                )

                # -------------------------------------------------
                # ESG GMV weight table
                # -------------------------------------------------
                st.subheader("ESG-screened minimum-variance portfolio weights")
                esg_gmv_weights = weights_to_table(
                    weights=gmv_esg_firms["Weights"],
                    firms=firms_esg,
                    label="ESG-screened GMV",
                )

                st.dataframe(
                    format_weight_table(esg_gmv_weights.head(20)),
                    use_container_width=True,
                    height=500,
                )

                st.caption(
                    "The table shows the 20 largest absolute portfolio weights for the ESG-screened minimum-variance portfolio."
                )

                st.subheader("Top ESG-eligible firms by ESG score")
                eligible_display = firms_esg.sort_values("ESG Score", ascending=False).copy()
                st.dataframe(
                    eligible_display[["Ticker", "Company", "ISIN", "ESG Grade", "ESG Score (%)", "Expected Return (%)", "Annual Volatility (%)"]].head(20),
                    use_container_width=True,
                )

                # Downloads
                weights_csv = esg_gmv_weights.to_csv(index=False).encode("utf-8")
                frontier_csv = frontier_esg.to_csv(index=False).encode("utf-8")
                eligible_csv = firms_esg.to_csv(index=False).encode("utf-8")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        "Download ESG GMV weights CSV",
                        data=weights_csv,
                        file_name="esg_screened_gmv_weights.csv",
                        mime="text/csv",
                    )
                with c2:
                    st.download_button(
                        "Download ESG frontier CSV",
                        data=frontier_csv,
                        file_name="esg_screened_frontier.csv",
                        mime="text/csv",
                    )
                with c3:
                    st.download_button(
                        "Download ESG-eligible firms CSV",
                        data=eligible_csv,
                        file_name="esg_eligible_firms.csv",
                        mime="text/csv",
                    )

                with st.expander("Show all ESG-screened GMV weights"):
                    st.dataframe(format_weight_table(esg_gmv_weights), use_container_width=True, height=650)

                with st.expander("Show all usable firms in the real-stock universe"):
                    st.dataframe(
                        firms_all[["Ticker", "Company", "ISIN", "ESG Grade", "ESG Score (%)", "Expected Return (%)", "Annual Volatility (%)"]],
                        use_container_width=True,
                        height=500,
                    )

                with st.expander("Show correlation matrix for ESG-eligible firms"):
                    corr_esg = corr_sub.loc[firms_esg["Ticker"], firms_esg["Ticker"]]
                    st.dataframe(corr_esg, use_container_width=True, height=450)

        except Exception as e:
            st.error(f"Could not build the firm-level frontier from the uploaded files: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")
