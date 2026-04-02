import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="ESG Company Recommender", layout="wide")

# --------------------------------------------------
# Session state
# --------------------------------------------------
DEFAULTS = {
    "page": "intro",
    "min_esg_score": 50.0,
    "target_esg_score": 70.0,
    "lambda_esg": 0.70,
    "top_n": 10,
    "company_search": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data
def load_esg_data():
    candidate_paths = [
        Path("esg_data_2025_esgcombined_only.csv"),
    ]

    file_path = None
    for path in candidate_paths:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        raise FileNotFoundError(
            "Could not find 'esg_data_2025_esgcombined_only.csv'. "
            "Place it in the same folder as app.py or inside a 'data/' folder."
        )

    df = pd.read_csv(file_path)

    required_cols = {"comname", "value", "valuescore", "year", "fieldname"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only 2025 ESGCombinedScore rows
    df = df[(df["year"] == 2025) & (df["fieldname"] == "ESGCombinedScore")].copy()

    # Clean and rename
    df = df[["comname", "value", "valuescore"]].copy()
    df = df.rename(columns={
        "comname": "Company",
        "value": "ESG Grade",
        "valuescore": "ESG Score Raw"
    })

    df = df.dropna(subset=["Company", "ESG Score Raw"])

    # Convert score to 0-100 and 0-1 versions
    df["ESG Score (%)"] = df["ESG Score Raw"] * 100
    df["ESG Score (0-1)"] = df["ESG Score Raw"]

    # Sort once for stability
    df = df.sort_values("Company").reset_index(drop=True)

    return df


def score_companies(df: pd.DataFrame, min_esg_score: float, target_esg_score: float, lambda_esg: float, company_search: str):
    """
    Preference score:
    - lambda_esg weights 'how high' the ESG score is
    - (1-lambda_esg) weights 'how close' the company is to the user's target ESG score

    preference_score = λ * ESG_level + (1-λ) * closeness_to_target
    """
    result = df.copy()

    # Optional company search
    if company_search.strip():
        result = result[result["Company"].str.contains(company_search.strip(), case=False, na=False)].copy()

    # Minimum ESG threshold
    result = result[result["ESG Score (%)"] >= min_esg_score].copy()

    if result.empty:
        return result

    # Preference components
    result["Closeness To Target"] = 1 - (result["ESG Score (%)"] - target_esg_score).abs() / 100
    result["Closeness To Target"] = result["Closeness To Target"].clip(lower=0)

    result["Preference Utility"] = (
        lambda_esg * result["ESG Score (0-1)"]
        + (1 - lambda_esg) * result["Closeness To Target"]
    )

    result = result.sort_values(
        by=["Preference Utility", "ESG Score (%)"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return result


# --------------------------------------------------
# Page 1: Intro
# --------------------------------------------------
if st.session_state.page == "intro":
    st.title("ESG Company Recommender")

    st.markdown(
        """
        This app uses your uploaded file:

        **`esg_data_2025_esgcombined_only.csv`**

        It only looks at:
        - **2025**
        - **ESGCombinedScore**
        - each company's:
          - name
          - ESG letter grade
          - numeric ESG score

        The app then recommends companies based on the investor's ESG preferences.

        ### How the recommendation works
        Each company is ranked using a preference score:

        **Preference Utility = λ × ESG Score + (1 − λ) × Closeness to Target ESG**

        where:
        - **λ** controls how much the investor prioritises a higher ESG score
        - **Target ESG** is the score the investor ideally wants
        - **Minimum ESG** filters out companies below the acceptable level
        """
    )

    if st.button("Continue"):
        go_to("inputs")


# --------------------------------------------------
# Page 2: Inputs
# --------------------------------------------------
elif st.session_state.page == "inputs":
    st.title("Investor ESG Preferences")

    with st.form("preferences_form"):
        st.subheader("Preference inputs")

        min_esg_score = st.number_input(
            "Minimum acceptable ESG score (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.min_esg_score),
            step=1.0,
            format="%.1f",
        )

        target_esg_score = st.number_input(
            "Target ESG score (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.target_esg_score),
            step=1.0,
            format="%.1f",
        )

        lambda_esg = st.slider(
            "ESG preference λ (1 = prioritise highest ESG score, 0 = prioritise closeness to your target)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.lambda_esg),
            step=0.01,
        )

        top_n = st.slider(
            "Number of recommendations to show",
            min_value=1,
            max_value=25,
            value=int(st.session_state.top_n),
            step=1,
        )

        company_search = st.text_input(
            "Optional company name filter",
            value=st.session_state.company_search
        )

        submitted = st.form_submit_button("Continue to results")

    if submitted:
        st.session_state.min_esg_score = min_esg_score
        st.session_state.target_esg_score = target_esg_score
        st.session_state.lambda_esg = lambda_esg
        st.session_state.top_n = top_n
        st.session_state.company_search = company_search
        go_to("results")

    if st.button("Back to introduction"):
        go_to("intro")


# --------------------------------------------------
# Page 3: Results
# --------------------------------------------------
elif st.session_state.page == "results":
    st.title("Results")

    try:
        df = load_esg_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    ranked = score_companies(
        df=df,
        min_esg_score=st.session_state.min_esg_score,
        target_esg_score=st.session_state.target_esg_score,
        lambda_esg=st.session_state.lambda_esg,
        company_search=st.session_state.company_search,
    )

    st.markdown(
        f"""
        **Dataset summary**
        - Companies available: **{len(df)}**
        - Minimum acceptable ESG score: **{st.session_state.min_esg_score:.1f}%**
        - Target ESG score: **{st.session_state.target_esg_score:.1f}%**
        - ESG preference λ: **{st.session_state.lambda_esg:.2f}**
        """
    )

    if ranked.empty:
        st.warning("No companies match your filters. Try lowering the minimum ESG score or clearing the company search.")
    else:
        best = ranked.iloc[0]
        top_df = ranked.head(st.session_state.top_n).copy()

        st.subheader("Best match")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Company", best["Company"])
        col2.metric("ESG Grade", best["ESG Grade"])
        col3.metric("ESG Score", f"{best['ESG Score (%)']:.1f}%")
        col4.metric("Preference Utility", f"{best['Preference Utility']:.3f}")

        st.subheader("Summary table")

        summary = pd.DataFrame([
            {
                "Recommended Company": best["Company"],
                "ESG Grade": best["ESG Grade"],
                "ESG Score (%)": best["ESG Score (%)"],
                "Closeness To Target": best["Closeness To Target"],
                "Preference Utility": best["Preference Utility"],
            }
        ])

        st.dataframe(
            summary.style.format({
                "ESG Score (%)": "{:.1f}",
                "Closeness To Target": "{:.3f}",
                "Preference Utility": "{:.3f}",
            }),
            use_container_width=True
        )

        st.subheader(f"Top {st.session_state.top_n} recommendations")

        st.dataframe(
            top_df[[
                "Company",
                "ESG Grade",
                "ESG Score (%)",
                "Closeness To Target",
                "Preference Utility"
            ]].style.format({
                "ESG Score (%)": "{:.1f}",
                "Closeness To Target": "{:.3f}",
                "Preference Utility": "{:.3f}",
            }),
            use_container_width=True
        )

        st.subheader("Recommendation chart")

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = top_df.sort_values("Preference Utility", ascending=True)

        ax.barh(plot_df["Company"], plot_df["Preference Utility"])
        ax.set_xlabel("Preference Utility")
        ax.set_ylabel("Company")
        ax.set_title("Top ESG Recommendations")

        st.pyplot(fig)

        st.subheader("ESG score distribution")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(df["ESG Score (%)"], bins=12)
        ax2.axvline(st.session_state.min_esg_score, linestyle="--", label="Minimum ESG")
        ax2.axvline(st.session_state.target_esg_score, linestyle=":", label="Target ESG")
        ax2.set_xlabel("ESG Score (%)")
        ax2.set_ylabel("Number of companies")
        ax2.set_title("2025 ESGCombinedScore distribution")
        ax2.legend()

        st.pyplot(fig2)

        download_cols = top_df[[
            "Company",
            "ESG Grade",
            "ESG Score (%)",
            "Closeness To Target",
            "Preference Utility"
        ]].copy()

        csv_data = download_cols.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download recommendations as CSV",
            data=csv_data,
            file_name="esg_recommendations_2025.csv",
            mime="text/csv"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to inputs"):
            go_to("inputs")
    with col2:
        if st.button("Start over"):
            go_to("intro")
