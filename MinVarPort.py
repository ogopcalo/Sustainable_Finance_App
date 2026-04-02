import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# =========================================================
# Finance / ESG functions
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

        # Utility from your slide:
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
# Page 1: Introduction
# =========================================================
if st.session_state.page == "intro":
    st.title("ESG Portfolio Optimiser")

    st.markdown(
        r"""
        This app compares:

        - a **standard mean-variance setup** using **all portfolios**
        - an **ESG-screened setup** using only portfolios that satisfy a **minimum portfolio ESG score**

        The investor utility is:

        \[
        U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda \bar{s}
        \]

        where:

        - \(E[R_p]\): expected portfolio return  
        - \(\sigma_p\): portfolio standard deviation  
        - \(\gamma\): risk aversion  
        - \(\bar{s}\): weighted average portfolio ESG score  
        - \(\lambda\): ESG preference intensity  

        In the ESG graph, the feasible set is reduced by a minimum ESG requirement, which can lower the
        tangency portfolio's Sharpe ratio and flatten the CML.
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
        **ESG screen used in Graph 2**  
        Required portfolio ESG score = **{esg_cutoff * 100:.2f} / 100**

        This threshold is implied by your ESG preference \( \lambda = {lambda_esg:.2f} \).
        """
    )

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

    # Optional full table
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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")def var_covar(sigmas, rho):
    return np.array([
        [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
        [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2]
    ])


def portfolio_stats(mu, sigma, rho, w1, rf):
    cov = var_covar(sigma, rho)
    w = np.array([w1, 1 - w1])

    port_return = np.dot(mu, w)
    port_variance = np.dot(w, np.dot(cov, w))
    port_std = np.sqrt(port_variance)

    sharpe = np.nan
    if port_std > 0:
        sharpe = (port_return - rf) / port_std

    return port_return, port_std, sharpe


def eff_front(mu, sigma, rho, weights, rf):
    returns = []
    std_devs = []
    sharpes = []

    for w1 in weights:
        ret, std, sharpe = portfolio_stats(mu, sigma, rho, w1, rf)
        returns.append(ret)
        std_devs.append(std)
        sharpes.append(sharpe)

    return np.array(returns), np.array(std_devs), np.array(sharpes)


def key_portfolios(mu, sigma, rho, weights, rf):
    returns, std_devs, sharpes = eff_front(mu, sigma, rho, weights, rf)

    min_var_idx = np.argmin(std_devs)
    max_sharpe_idx = np.nanargmax(sharpes)

    summary = {
        "Minimum Variance": {
            "Weight Asset 1": weights[min_var_idx],
            "Weight Asset 2": 1 - weights[min_var_idx],
            "Expected Return": returns[min_var_idx],
            "Standard Deviation": std_devs[min_var_idx],
            "Sharpe Ratio": sharpes[min_var_idx],
        },
        "Maximum Sharpe": {
            "Weight Asset 1": weights[max_sharpe_idx],
            "Weight Asset 2": 1 - weights[max_sharpe_idx],
            "Expected Return": returns[max_sharpe_idx],
            "Standard Deviation": std_devs[max_sharpe_idx],
            "Sharpe Ratio": sharpes[max_sharpe_idx],
        },
    }

    return summary, returns, std_devs, sharpes


# -----------------------------
# Page 1: Introduction
# -----------------------------
if st.session_state.page == "intro":
    st.title("Efficient Frontier App")

    st.markdown(
        """
        Welcome to this portfolio analysis app.

        This tool lets you:
        - enter expected returns and volatilities for two assets,
        - choose portfolio assumptions such as correlation and the risk-free rate,
        - view the portfolio frontier,
        - identify the minimum-variance and maximum-Sharpe portfolios.

        Click below to continue to the input page.
        """
    )

    st.button("Continue", on_click=go_to, args=("inputs",))


# -----------------------------
# Page 2: Inputs
# -----------------------------
elif st.session_state.page == "inputs":
    st.title("Portfolio Inputs")

    st.write("Enter the assumptions for Asset 1, Asset 2, and the portfolio settings.")

    with st.form("portfolio_input_form"):
        st.subheader("Asset 1 inputs")
        mu1 = st.number_input(
            "Expected return for Asset 1",
            min_value=0.00,
            max_value=1.00,
            value=float(st.session_state.mu1),
            step=0.01
        )
        sigma1 = st.number_input(
            "Volatility for Asset 1",
            min_value=0.001,
            max_value=1.00,
            value=float(st.session_state.sigma1),
            step=0.01
        )

        st.markdown("---")

        st.subheader("Asset 2 inputs")
        mu2 = st.number_input(
            "Expected return for Asset 2",
            min_value=0.00,
            max_value=1.00,
            value=float(st.session_state.mu2),
            step=0.01
        )
        sigma2 = st.number_input(
            "Volatility for Asset 2",
            min_value=0.001,
            max_value=1.00,
            value=float(st.session_state.sigma2),
            step=0.01
        )

        st.markdown("---")

        st.subheader("Portfolio inputs")
        rf = st.number_input(
            "Risk-free rate",
            min_value=0.00,
            max_value=1.00,
            value=float(st.session_state.rf),
            step=0.005
        )

        available_rhos = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]

        selected_rhos = st.multiselect(
            "Correlations to plot",
            options=available_rhos,
            default=st.session_state.selected_rhos
        )

        if not selected_rhos:
            selected_rhos = [0.0]

        detailed_rho = st.selectbox(
            "Correlation to use for the summary results",
            options=selected_rhos,
            index=selected_rhos.index(st.session_state.detailed_rho)
            if st.session_state.detailed_rho in selected_rhos else 0
        )

        num_points = st.slider(
            "Number of weight points",
            min_value=101,
            max_value=5001,
            value=int(st.session_state.num_points),
            step=100
        )

        submitted = st.form_submit_button("Continue to results")

        if submitted:
            st.session_state.mu1 = mu1
            st.session_state.mu2 = mu2
            st.session_state.sigma1 = sigma1
            st.session_state.sigma2 = sigma2
            st.session_state.rf = rf
            st.session_state.selected_rhos = selected_rhos
            st.session_state.detailed_rho = detailed_rho
            st.session_state.num_points = num_points
            st.session_state.page = "results"
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to introduction"):
            go_to("intro")


# -----------------------------
# Page 3: Results
# -----------------------------
elif st.session_state.page == "results":
    st.title("Results")

    mu = np.array([st.session_state.mu1, st.session_state.mu2])
    sigma = np.array([st.session_state.sigma1, st.session_state.sigma2])
    rf = st.session_state.rf
    selected_rhos = st.session_state.selected_rhos
    detailed_rho = st.session_state.detailed_rho
    weights = np.linspace(0, 1, st.session_state.num_points)

    summary, returns, std_devs, sharpes = key_portfolios(
        mu, sigma, detailed_rho, weights, rf
    )

    st.subheader("Efficient frontier chart")

    fig, ax = plt.subplots(figsize=(10, 6))

    for rho in selected_rhos:
        curve_returns, curve_stds, _ = eff_front(mu, sigma, rho, weights, rf)
        ax.plot(curve_stds, curve_returns, label=f"ρ = {rho}")

    ax.scatter(
        summary["Minimum Variance"]["Standard Deviation"],
        summary["Minimum Variance"]["Expected Return"],
        s=90,
        marker="o",
        label=f"Min-Variance (ρ={detailed_rho})"
    )

    ax.scatter(
        summary["Maximum Sharpe"]["Standard Deviation"],
        summary["Maximum Sharpe"]["Expected Return"],
        s=180,
        marker="*",
        label=f"Max-Sharpe (ρ={detailed_rho})"
    )

    ax.set_xlabel("Portfolio Standard Deviation")
    ax.set_ylabel("Portfolio Expected Return")
    ax.set_title("Portfolio Frontier")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    st.subheader("Summary table")

    summary_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "Portfolio"})
    st.dataframe(
        summary_df.style.format({
            "Weight Asset 1": "{:.2%}",
            "Weight Asset 2": "{:.2%}",
            "Expected Return": "{:.2%}",
            "Standard Deviation": "{:.2%}",
            "Sharpe Ratio": "{:.3f}",
        }),
        use_container_width=True
    )

    st.subheader("Full frontier table")

    frontier_df = pd.DataFrame({
        "Weight Asset 1": weights,
        "Weight Asset 2": 1 - weights,
        "Expected Return": returns,
        "Standard Deviation": std_devs,
        "Sharpe Ratio": sharpes
    })

    st.dataframe(
        frontier_df.style.format({
            "Weight Asset 1": "{:.2%}",
            "Weight Asset 2": "{:.2%}",
            "Expected Return": "{:.2%}",
            "Standard Deviation": "{:.2%}",
            "Sharpe Ratio": "{:.3f}",
        }),
        use_container_width=True,
        height=350
    )

    csv_data = frontier_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download frontier table as CSV",
        data=csv_data,
        file_name="frontier_results.csv",
        mime="text/csv"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")
detailed_rho = st.sidebar.selectbox(
    "Correlation for detailed portfolio metrics",
    options=selected_rhos,
    index=0
)

num_points = st.sidebar.slider("Number of weight points", 101, 5001, 1001, 100)

mu = np.array([mu1, mu2])
sigma = np.array([sigma1, sigma2])
weights = np.linspace(0, 1, num_points)

# -----------------------------
# Functions
# -----------------------------
def var_covar(sigma, rho):
    """Variance-covariance matrix for two assets."""
    return np.array([
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2]
    ])


def portfolio_stats(mu, sigma, rho, w1, rf):
    """Return portfolio return, std dev, and Sharpe ratio."""
    cov = var_covar(sigma, rho)
    w = np.array([w1, 1 - w1])

    port_return = np.dot(mu, w)
    port_variance = np.dot(w, np.dot(cov, w))
    port_std = np.sqrt(port_variance)

    if port_std > 0:
        sharpe = (port_return - rf) / port_std
    else:
        sharpe = np.nan

    return port_return, port_std, sharpe


def eff_front(mu, sigma, rho, weights, rf):
    """Compute frontier values over a grid of portfolio weights."""
    returns = []
    std_devs = []
    sharpes = []

    for w1 in weights:
        ret_p, std_p, sharpe_p = portfolio_stats(mu, sigma, rho, w1, rf)
        returns.append(ret_p)
        std_devs.append(std_p)
        sharpes.append(sharpe_p)

    return np.array(returns), np.array(std_devs), np.array(sharpes)


def key_portfolios(mu, sigma, rho, weights, rf):
    """Find min-variance and max-Sharpe portfolios."""
    returns, std_devs, sharpes = eff_front(mu, sigma, rho, weights, rf)

    min_var_idx = np.argmin(std_devs)
    max_sharpe_idx = np.nanargmax(sharpes)

    results = {
        "min_var": {
            "w1": weights[min_var_idx],
            "w2": 1 - weights[min_var_idx],
            "return": returns[min_var_idx],
            "std": std_devs[min_var_idx],
            "sharpe": sharpes[min_var_idx]
        },
        "max_sharpe": {
            "w1": weights[max_sharpe_idx],
            "w2": 1 - weights[max_sharpe_idx],
            "return": returns[max_sharpe_idx],
            "std": std_devs[max_sharpe_idx],
            "sharpe": sharpes[max_sharpe_idx]
        }
    }

    return results, returns, std_devs, sharpes


# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    with _lock:
        fig, ax = plt.subplots(figsize=(10, 6))

        for rho in selected_rhos:
            returns, std_devs, _ = eff_front(mu, sigma, rho, weights, rf)
            ax.plot(std_devs, returns, label=f"ρ = {rho}")

        # highlight detailed rho portfolios
        detail_results, detail_returns, detail_stds, detail_sharpes = key_portfolios(
            mu, sigma, detailed_rho, weights, rf
        )

        ax.scatter(
            detail_results["min_var"]["std"],
            detail_results["min_var"]["return"],
            marker="o",
            s=80,
            label=f"Min-Var (ρ={detailed_rho})"
        )

        ax.scatter(
            detail_results["max_sharpe"]["std"],
            detail_results["max_sharpe"]["return"],
            marker="*",
            s=180,
            label=f"Max-Sharpe (ρ={detailed_rho})"
        )

        ax.set_xlabel("Portfolio Standard Deviation")
        ax.set_ylabel("Portfolio Expected Return")
        ax.set_title("Portfolio Frontier for Different Correlations")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

with col2:
    st.subheader("Current inputs")
    st.write(f"**Asset 1 expected return:** {mu1:.2%}")
    st.write(f"**Asset 2 expected return:** {mu2:.2%}")
    st.write(f"**Asset 1 volatility:** {sigma1:.2%}")
    st.write(f"**Asset 2 volatility:** {sigma2:.2%}")
    st.write(f"**Risk-free rate:** {rf:.2%}")
    st.write(f"**Detailed correlation:** {detailed_rho}")

# -----------------------------
# Detailed metrics table
# -----------------------------
results, returns, std_devs, sharpes = key_portfolios(mu, sigma, detailed_rho, weights, rf)

summary_df = pd.DataFrame([
    {
        "Portfolio": "Minimum Variance",
        "Weight Asset 1": results["min_var"]["w1"],
        "Weight Asset 2": results["min_var"]["w2"],
        "Expected Return": results["min_var"]["return"],
        "Standard Deviation": results["min_var"]["std"],
        "Sharpe Ratio": results["min_var"]["sharpe"]
    },
    {
        "Portfolio": "Maximum Sharpe",
        "Weight Asset 1": results["max_sharpe"]["w1"],
        "Weight Asset 2": results["max_sharpe"]["w2"],
        "Expected Return": results["max_sharpe"]["return"],
        "Standard Deviation": results["max_sharpe"]["std"],
        "Sharpe Ratio": results["max_sharpe"]["sharpe"]
    }
])

st.subheader(f"Detailed portfolio metrics for ρ = {detailed_rho}")
st.dataframe(
    summary_df.style.format({
        "Weight Asset 1": "{:.2%}",
        "Weight Asset 2": "{:.2%}",
        "Expected Return": "{:.2%}",
        "Standard Deviation": "{:.2%}",
        "Sharpe Ratio": "{:.3f}"
    }),
    use_container_width=True
)

# -----------------------------
# Full frontier data table
# -----------------------------
frontier_df = pd.DataFrame({
    "Weight Asset 1": weights,
    "Weight Asset 2": 1 - weights,
    "Expected Return": returns,
    "Standard Deviation": std_devs,
    "Sharpe Ratio": sharpes
})

st.subheader("Frontier data")
st.dataframe(
    frontier_df.style.format({
        "Weight Asset 1": "{:.2%}",
        "Weight Asset 2": "{:.2%}",
        "Expected Return": "{:.2%}",
        "Standard Deviation": "{:.2%}",
        "Sharpe Ratio": "{:.3f}"
    }),
    use_container_width=True,
    height=350
)

csv_data = frontier_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download frontier data as CSV",
    data=csv_data,
    file_name=f"frontier_rho_{detailed_rho}.csv",
    mime="text/csv"
)

# -----------------------------
# Optional explanation
# -----------------------------
with st.expander("What this app shows"):
    st.markdown(
        """
        - **Expected return** is the weighted average of the two assets' expected returns.
        - **Portfolio risk** depends on both individual volatilities and the correlation between assets.
        - **Minimum-variance portfolio** is the portfolio with the lowest standard deviation.
        - **Maximum-Sharpe portfolio** is the portfolio with the highest risk-adjusted return:
        
        \[
        \text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
        \]
        """
    )
