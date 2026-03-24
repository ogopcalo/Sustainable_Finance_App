import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(page_title="ESG Portfolio App", layout="wide")

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
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


def var_covar(sigmas: np.ndarray, rho: float) -> np.ndarray:
    return np.array([
        [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
        [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2]
    ])


def build_portfolio_grid(mu, sigma, rho, rf, esg_scores, lambda_esg, gamma, num_points):
    cov = var_covar(sigma, rho)
    weights = np.linspace(0, 1, num_points)

    rows = []
    for w1 in weights:
        w = np.array([w1, 1 - w1])

        exp_return = np.dot(mu, w)
        variance = np.dot(w, np.dot(cov, w))
        std_dev = np.sqrt(max(variance, 0.0))
        esg_score = np.dot(esg_scores, w)

        esg_adjusted_return = exp_return + lambda_esg * esg_score

        financial_sharpe = np.nan if std_dev == 0 else (exp_return - rf) / std_dev
        esg_adjusted_sharpe = np.nan if std_dev == 0 else (esg_adjusted_return - rf) / std_dev
        utility = exp_return - 0.5 * gamma * (std_dev ** 2) + lambda_esg * esg_score

        rows.append({
            "Weight Asset 1": w1,
            "Weight Asset 2": 1 - w1,
            "Expected Return": exp_return,
            "ESG-Adjusted Return": esg_adjusted_return,
            "Std Dev": std_dev,
            "Variance": variance,
            "ESG Score": esg_score,
            "Financial Sharpe": financial_sharpe,
            "ESG-Adjusted Sharpe": esg_adjusted_sharpe,
            "Utility": utility,
        })

    return pd.DataFrame(rows)


def select_key_portfolios(df: pd.DataFrame, lambda_esg: float):
    idx_tan_std = df["Financial Sharpe"].idxmax()
    tan_std = df.loc[idx_tan_std]

    idx_tan_esg = df["ESG-Adjusted Sharpe"].idxmax()
    tan_esg = df.loc[idx_tan_esg]

    idx_mvp = df["Std Dev"].idxmin()
    mvp = df.loc[idx_mvp]

    esg_min = df["ESG Score"].min()
    esg_max = df["ESG Score"].max()
    esg_target = esg_min + lambda_esg * (esg_max - esg_min)

    feasible = df[df["ESG Score"] >= esg_target - 1e-12].copy()
    idx_mvp_esg = feasible["Std Dev"].idxmin()
    mvp_esg = feasible.loc[idx_mvp_esg]

    idx_u = df["Utility"].idxmax()
    u_max = df.loc[idx_u]

    return {
        "tan_std": tan_std,
        "tan_esg": tan_esg,
        "mvp": mvp,
        "mvp_esg": mvp_esg,
        "u_max": u_max,
        "esg_target": esg_target,
        "feasible_df": feasible,
    }


def summary_table(selected: dict) -> pd.DataFrame:
    rows = []

    def add_row(name: str, row: pd.Series):
        rows.append({
            "Portfolio": name,
            "Weight Asset 1": row["Weight Asset 1"],
            "Weight Asset 2": row["Weight Asset 2"],
            "Expected Return": row["Expected Return"],
            "ESG-Adjusted Return": row["ESG-Adjusted Return"],
            "Std Dev": row["Std Dev"],
            "ESG Score": row["ESG Score"],
            "Financial Sharpe": row["Financial Sharpe"],
            "ESG-Adjusted Sharpe": row["ESG-Adjusted Sharpe"],
            "Utility": row["Utility"],
        })

    add_row("Max Sharpe (Standard)", selected["tan_std"])
    add_row("Max Sharpe (ESG-Aware)", selected["tan_esg"])
    add_row("Min Variance (Standard)", selected["mvp"])
    add_row("Min Variance (ESG-Constrained)", selected["mvp_esg"])
    add_row("Max Utility", selected["u_max"])

    return pd.DataFrame(rows)


if st.session_state.page == "intro":
    st.title("ESG Portfolio Optimiser")

    st.markdown("""
    This app extends the standard portfolio utility:
    **U = E[Rₚ] − (γ/2)σₚ² + λs̄**

    You will input asset returns, volatilities, correlation, ESG scores,
    and the investor's ESG preference.
    """)

    if st.button("Continue"):
        go_to("inputs")

elif st.session_state.page == "inputs":
    st.title("Portfolio Inputs")

    with st.form("inputs_form"):
        st.subheader("Asset 1 inputs")
        mu1_pct = st.number_input("Expected return for Asset 1 (%)", 0.0, 100.0, float(st.session_state.mu1_pct), 0.25, format="%.2f")
        sigma1_pct = st.number_input("Volatility for Asset 1 (%)", 0.01, 100.0, float(st.session_state.sigma1_pct), 0.25, format="%.2f")
        esg1 = st.number_input("ESG score for Asset 1 (0 to 100)", 0.0, 100.0, float(st.session_state.esg1), 1.0, format="%.1f")

        st.markdown("---")

        st.subheader("Asset 2 inputs")
        mu2_pct = st.number_input("Expected return for Asset 2 (%)", 0.0, 100.0, float(st.session_state.mu2_pct), 0.25, format="%.2f")
        sigma2_pct = st.number_input("Volatility for Asset 2 (%)", 0.01, 100.0, float(st.session_state.sigma2_pct), 0.25, format="%.2f")
        esg2 = st.number_input("ESG score for Asset 2 (0 to 100)", 0.0, 100.0, float(st.session_state.esg2), 1.0, format="%.1f")

        st.markdown("---")

        st.subheader("Portfolio inputs")
        rf_pct = st.number_input("Risk-free rate (%)", 0.0, 100.0, float(st.session_state.rf_pct), 0.25, format="%.2f")
        rho = st.slider("Correlation between Asset 1 and Asset 2", -1.0, 1.0, float(st.session_state.rho), 0.01)
        num_points = st.slider("Number of portfolio weight points", 101, 5001, int(st.session_state.num_points), 100)

        st.markdown("---")

        st.subheader("Investor ESG preferences")
        lambda_esg = st.slider("ESG preference intensity λ", 0.0, 1.0, float(st.session_state.lambda_esg), 0.01)
        gamma = st.number_input("Risk aversion γ", 0.0, 50.0, float(st.session_state.gamma), 0.1, format="%.2f")

        submitted = st.form_submit_button("Continue to results")

    if submitted:
        st.session_state.mu1_pct = mu1_pct
        st.session_state.mu2_pct = mu2_pct
        st.session_state.sigma1_pct = sigma1_pct
        st.session_state.sigma2_pct = sigma2_pct
        st.session_state.rf_pct = rf_pct
        st.session_state.rho = rho
        st.session_state.num_points = num_points
        st.session_state.esg1 = esg1
        st.session_state.esg2 = esg2
        st.session_state.lambda_esg = lambda_esg
        st.session_state.gamma = gamma
        go_to("results")

    if st.button("Back to introduction"):
        go_to("intro")

elif st.session_state.page == "results":
    st.title("Results")

    mu = np.array([st.session_state.mu1_pct, st.session_state.mu2_pct]) / 100.0
    sigma = np.array([st.session_state.sigma1_pct, st.session_state.sigma2_pct]) / 100.0
    rf = st.session_state.rf_pct / 100.0
    rho = st.session_state.rho
    esg_scores = np.array([st.session_state.esg1, st.session_state.esg2]) / 100.0
    lambda_esg = st.session_state.lambda_esg
    gamma = st.session_state.gamma
    num_points = st.session_state.num_points

    df = build_portfolio_grid(mu, sigma, rho, rf, esg_scores, lambda_esg, gamma, num_points)
    selected = select_key_portfolios(df, lambda_esg)
    summ = summary_table(selected)

    st.subheader("1) Maximum Sharpe portfolio: with vs without ESG")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    tan_std = selected["tan_std"]
    tan_esg = selected["tan_esg"]

    ax1.plot(df["Std Dev"], df["Expected Return"], label="Mean-variance frontier (standard)")
    ax1.plot(df["Std Dev"], df["ESG-Adjusted Return"], linestyle="--", label="Mean-variance frontier (with ESG)")
    ax1.scatter([0], [rf], s=80, label="Risk-free asset")

    sigma_line_max = max(df["Std Dev"].max(), tan_std["Std Dev"], tan_esg["Std Dev"]) * 1.10
    sigma_line = np.linspace(0, sigma_line_max, 200)

    ax1.plot(sigma_line, rf + tan_std["Financial Sharpe"] * sigma_line, label="CML / CAL (standard)")
    ax1.plot(sigma_line, rf + tan_esg["ESG-Adjusted Sharpe"] * sigma_line, linestyle="--", label="CML / CAL (with ESG)")

    ax1.scatter([tan_std["Std Dev"]], [tan_std["Expected Return"]], s=120, marker="o", label="Max Sharpe (standard)")
    ax1.scatter([tan_esg["Std Dev"]], [tan_esg["ESG-Adjusted Return"]], s=160, marker="*", label="Max Sharpe (with ESG)")

    ax1.set_xlabel("Portfolio standard deviation")
    ax1.set_ylabel("Return / ESG-adjusted return")
    ax1.set_title("Maximum Sharpe comparison: standard vs ESG-adjusted")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("2) Minimum-variance portfolio: with vs without ESG")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    mvp = selected["mvp"]
    mvp_esg = selected["mvp_esg"]
    feasible_df = selected["feasible_df"].sort_values("Std Dev")

    ax2.plot(df["Std Dev"], df["Expected Return"], label="Mean-variance frontier (standard)")
    ax2.plot(df["Std Dev"], df["ESG-Adjusted Return"], linestyle="--", label="Mean-variance frontier (with ESG)")
    ax2.plot(feasible_df["Std Dev"], feasible_df["Expected Return"], linewidth=3, label=f"ESG-feasible set (score ≥ {selected['esg_target'] * 100:.1f})")

    ax2.scatter([mvp["Std Dev"]], [mvp["Expected Return"]], s=120, marker="o", label="Min variance (standard)")
    ax2.scatter([mvp_esg["Std Dev"]], [mvp_esg["ESG-Adjusted Return"]], s=160, marker="*", label="Min variance (with ESG)")

    ax2.set_xlabel("Portfolio standard deviation")
    ax2.set_ylabel("Return / ESG-adjusted return")
    ax2.set_title("Minimum-variance comparison: standard vs ESG-adjusted")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Summary table")
    st.dataframe(
        summ.style.format({
            "Weight Asset 1": "{:.2%}",
            "Weight Asset 2": "{:.2%}",
            "Expected Return": "{:.2%}",
            "ESG-Adjusted Return": "{:.2%}",
            "Std Dev": "{:.2%}",
            "ESG Score": "{:.2%}",
            "Financial Sharpe": "{:.3f}",
            "ESG-Adjusted Sharpe": "{:.3f}",
            "Utility": "{:.4f}",
        }),
        use_container_width=True
    )

    st.subheader("Full portfolio table")
    st.dataframe(
        df.style.format({
            "Weight Asset 1": "{:.2%}",
            "Weight Asset 2": "{:.2%}",
            "Expected Return": "{:.2%}",
            "ESG-Adjusted Return": "{:.2%}",
            "Std Dev": "{:.2%}",
            "Variance": "{:.5f}",
            "ESG Score": "{:.2%}",
            "Financial Sharpe": "{:.3f}",
            "ESG-Adjusted Sharpe": "{:.3f}",
            "Utility": "{:.4f}",
        }),
        use_container_width=True,
        height=350
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full portfolio table as CSV", csv_data, "esg_portfolio_results.csv", "text/csv")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")
