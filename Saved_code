import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(page_title="Efficient Frontier App", layout="wide")


# -----------------------------
# Session state defaults
# -----------------------------
defaults = {
    "page": "intro",
    "mu1": 0.05,
    "mu2": 0.12,
    "sigma1": 0.09,
    "sigma2": 0.20,
    "rf": 0.02,
    "selected_rhos": [-1.0, -0.2, 0.0],
    "detailed_rho": -0.2,
    "num_points": 1001,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# -----------------------------
# Finance functions
# -----------------------------
def var_covar(sigmas, rho):
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
