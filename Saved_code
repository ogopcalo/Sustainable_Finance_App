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
# Company ESG data functions
# =========================================================
@st.cache_data
def load_company_esg_data() -> pd.DataFrame:
    """
    Load the 2025 ESGCombinedScore company file from repo/local environment.
    Expected raw columns from your CSV:
      comname, value, valuescore, year, fieldname, ...
    """
    candidate_paths = [
        Path("esg_data_2025_esgcombined_only.csv"),
        Path("/mnt/esg_data_2025_esgcombined_only.csv"),
    ]

    file_path = None
    for p in candidate_paths:
        if p.exists():
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError(
            "Could not find 'esg_data_2025_esgcombined_only.csv'. "
            "Put it in the same folder as app.py or in a data/ folder."
        )

    raw = pd.read_csv(file_path)

    required_cols = {"comname", "value", "valuescore", "year", "fieldname"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = raw[(raw["year"] == 2025) & (raw["fieldname"] == "ESGCombinedScore")].copy()

    keep_cols = ["comname", "value", "valuescore"]
    if "ticker" in df.columns:
        keep_cols.append("ticker")
    if "isin" in df.columns:
        keep_cols.append("isin")

    df = df[keep_cols].copy()

    rename_map = {
        "comname": "Company",
        "value": "ESG Grade",
        "valuescore": "ESG Score",
        "ticker": "Ticker",
        "isin": "ISIN",
    }
    df = df.rename(columns=rename_map)

    df["ESG Score"] = pd.to_numeric(df["ESG Score"], errors="coerce")
    df = df.dropna(subset=["Company", "ESG Score"]).copy()

    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    if "ISIN" not in df.columns:
        df["ISIN"] = ""

    df["ESG Score (%)"] = df["ESG Score"] * 100
    df = df.sort_values("Company").reset_index(drop=True)
    return df


def recommend_firm_pair_for_esg_mvp(
    company_df: pd.DataFrame,
    portfolio_weights: tuple[float, float],
    target_asset_esg: tuple[float, float],
    target_portfolio_esg: float,
    min_company_esg: float,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """
    Recommend two firms from the ESG dataset for the ESG minimum-variance portfolio.

    Since the file only contains ESG scores, this is NOT a true firm-level MVP.
    Instead, it selects two firms that:
    - pass the company ESG threshold
    - best match the portfolio ESG target and the entered asset ESG profile
    - use the model's ESG-MVP weights
    """
    eligible = company_df[company_df["ESG Score"] >= min_company_esg].copy()

    if len(eligible) < 2:
        return None, eligible

    w1, w2 = portfolio_weights
    target1, target2 = target_asset_esg

    rows = []
    for i, firm1 in eligible.iterrows():
        for j, firm2 in eligible.iterrows():
            if i == j:
                continue

            pair_portfolio_esg = w1 * firm1["ESG Score"] + w2 * firm2["ESG Score"]

            # Lower objective is better
            objective = (
                3.0 * abs(pair_portfolio_esg - target_portfolio_esg)
                + 1.0 * abs(firm1["ESG Score"] - target1)
                + 1.0 * abs(firm2["ESG Score"] - target2)
            )

            rows.append({
                "Firm 1": firm1["Company"],
                "Firm 1 Ticker": firm1.get("Ticker", ""),
                "Firm 1 ISIN": firm1.get("ISIN", ""),
                "Firm 1 ESG Grade": firm1["ESG Grade"],
                "Firm 1 ESG Score": firm1["ESG Score"],
                "Firm 1 Weight": w1,
                "Firm 2": firm2["Company"],
                "Firm 2 Ticker": firm2.get("Ticker", ""),
                "Firm 2 ISIN": firm2.get("ISIN", ""),
                "Firm 2 ESG Grade": firm2["ESG Grade"],
                "Firm 2 ESG Score": firm2["ESG Score"],
                "Firm 2 Weight": w2,
                "Implied Portfolio ESG": pair_portfolio_esg,
                "Target Portfolio ESG": target_portfolio_esg,
                "Match Objective": objective,
            })

    pair_df = pd.DataFrame(rows).sort_values("Match Objective").reset_index(drop=True)
    best_pair = pair_df.iloc[0]
    return best_pair, eligible


def format_recommendation_table(df: pd.DataFrame):
    return df.style.format({
        "Firm 1 ESG Score": "{:.2%}",
        "Firm 2 ESG Score": "{:.2%}",
        "Firm 1 Weight": "{:.2%}",
        "Firm 2 Weight": "{:.2%}",
        "Implied Portfolio ESG": "{:.2%}",
        "Target Portfolio ESG": "{:.2%}",
        "Match Objective": "{:.4f}",
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

    tab_analysis, tab_reco = st.tabs(["Portfolio analysis", "Firm recommendation"])

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

    with tab_reco:
        st.subheader("Firm recommendation for the ESG minimum-variance portfolio")

        st.markdown(
            """
            This tab maps your **ESG minimum-variance portfolio** onto **2 firms** from
            `esg_data_2025_esgcombined_only.csv`.

            The selected firms:
            - come from the 2025 ESG dataset
            - satisfy a minimum company ESG score
            - best match the ESG profile of your ESG-screened minimum-variance portfolio

            **Note:** this is an ESG-only matching step.  
            The dataset does not contain firm-level returns, volatilities, or correlations,
            so this is not a true firm-level minimum-variance optimisation.
            """
        )

        try:
            company_df = load_company_esg_data()

            min_company_esg_pct = st.number_input(
                "Minimum company ESG score for recommendation (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(esg_cutoff * 100),
                step=1.0,
                format="%.1f",
                key="min_company_esg_pct_reco"
            )

            best_pair, eligible_companies = recommend_firm_pair_for_esg_mvp(
                company_df=company_df,
                portfolio_weights=(float(mvp_esg["Weight Asset 1"]), float(mvp_esg["Weight Asset 2"])),
                target_asset_esg=(st.session_state.esg1 / 100.0, st.session_state.esg2 / 100.0),
                target_portfolio_esg=float(mvp_esg["ESG Score"]),
                min_company_esg=min_company_esg_pct / 100.0,
            )

            st.write(f"Companies available in 2025 ESG file: **{len(company_df)}**")
            st.write(f"Companies meeting the company ESG cutoff: **{len(eligible_companies)}**")

            if best_pair is None:
                st.warning("Fewer than 2 firms meet the company ESG cutoff. Lower the minimum company ESG score.")
            else:
                st.subheader("Recommended 2-firm ESG minimum-variance mapping")

                reco_summary = pd.DataFrame([
                    {
                        "Firm 1": best_pair["Firm 1"],
                        "Firm 1 Ticker": best_pair["Firm 1 Ticker"],
                        "Firm 1 ISIN": best_pair["Firm 1 ISIN"],
                        "Firm 1 ESG Grade": best_pair["Firm 1 ESG Grade"],
                        "Firm 1 ESG Score": best_pair["Firm 1 ESG Score"],
                        "Firm 1 Weight": best_pair["Firm 1 Weight"],
                        "Firm 2": best_pair["Firm 2"],
                        "Firm 2 Ticker": best_pair["Firm 2 Ticker"],
                        "Firm 2 ISIN": best_pair["Firm 2 ISIN"],
                        "Firm 2 ESG Grade": best_pair["Firm 2 ESG Grade"],
                        "Firm 2 ESG Score": best_pair["Firm 2 ESG Score"],
                        "Firm 2 Weight": best_pair["Firm 2 Weight"],
                        "Implied Portfolio ESG": best_pair["Implied Portfolio ESG"],
                        "Target Portfolio ESG": best_pair["Target Portfolio ESG"],
                        "Match Objective": best_pair["Match Objective"],
                    }
                ])

                st.dataframe(format_recommendation_table(reco_summary), use_container_width=True)

                detail_df = pd.DataFrame([
                    {
                        "Portfolio being matched": "ESG Minimum Variance Portfolio",
                        "Weight Asset 1": mvp_esg["Weight Asset 1"],
                        "Weight Asset 2": mvp_esg["Weight Asset 2"],
                        "Target ESG Score": mvp_esg["ESG Score"],
                        "Utility": mvp_esg["Utility"],
                    }
                ])

                st.subheader("Model portfolio being mapped")
                st.dataframe(
                    detail_df.style.format({
                        "Weight Asset 1": "{:.2%}",
                        "Weight Asset 2": "{:.2%}",
                        "Target ESG Score": "{:.2%}",
                        "Utility": "{:.4f}",
                    }),
                    use_container_width=True
                )

                st.subheader("Recommended firm weights")
                weight_chart = pd.DataFrame({
                    "Firm": [best_pair["Firm 1"], best_pair["Firm 2"]],
                    "Weight": [best_pair["Firm 1 Weight"], best_pair["Firm 2 Weight"]],
                })

                fig3, ax3 = plt.subplots(figsize=(8, 4))
                ax3.bar(weight_chart["Firm"], weight_chart["Weight"])
                ax3.set_ylim(0, 1)
                ax3.set_ylabel("Portfolio weight")
                ax3.set_title("Recommended 2-firm weight split")
                st.pyplot(fig3)

                st.subheader("Top eligible firms by ESG score")
                eligible_display = eligible_companies.sort_values("ESG Score", ascending=False).head(15).copy()
                st.dataframe(
                    eligible_display[["Company", "Ticker", "ISIN", "ESG Grade", "ESG Score (%)"]].style.format({
                        "ESG Score (%)": "{:.1f}"
                    }),
                    use_container_width=True
                )

                reco_csv = reco_summary.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download recommendation as CSV",
                    data=reco_csv,
                    file_name="esg_mvp_firm_recommendation.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not load the ESG company file: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")
