"""
app.py
------
Phase 2 & 3: Streamlit UI for Customer Churn Prediction.
  - Hero section with clean branding
  - Sidebar for user inputs
  - EDA tab with interactive Plotly charts
  - Prediction tab with st.metric cards
  - Help / explainability section
  - Full error handling
"""

import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
META_PATH  = BASE_DIR / "model_meta.json"
DATA_PATH  = BASE_DIR / "customer_churn.csv"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Global typography --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* --- Hero banner --- */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .hero-banner p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }

    /* --- Metric cards --- */
    div[data-testid="metric-container"] {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }

    /* --- Prediction result boxes --- */
    .churn-box {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .safe-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.4rem; }
    .result-sub   { font-size: 0.95rem; color: #374151; }

    /* --- Predict button --- */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }

    /* --- Section headers --- */
    .section-header {
        font-size: 1.15rem; font-weight: 600;
        color: #0f172a; border-bottom: 2px solid #0ea5e9;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem 0;
    }
    /* --- Sidebar --- */
    section[data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    meta  = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return model, meta


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def build_input_df(inputs: dict, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the training schema."""
    row = {col: inputs.get(col, np.nan) for col in numeric_cols + categorical_cols}
    df  = pd.DataFrame([row])
    # Derived feature used in training
    df["Avg_Charge_Per_Month"] = (
        df["Total_Charges"] / df["Tenure_Months"].replace(0, 1)
    ).round(2)
    return df


def risk_label(prob: float) -> str:
    if prob < 0.35:  return "🟢 Low Risk"
    if prob < 0.60:  return "🟡 Moderate Risk"
    return "🔴 High Risk"


# ── Load artefacts ────────────────────────────────────────────────────────────
model, meta = load_model()
df_data     = load_dataset()

numeric_cols     = meta.get("numeric_cols",     []) if meta else []
categorical_cols = meta.get("categorical_cols", []) if meta else []
best_model_name  = meta.get("best_model", "Unknown") if meta else "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <h1>🛡️ ChurnGuard AI</h1>
  <p>
    Predict customer churn before it happens.
    Enter a customer's profile in the sidebar and get an instant risk assessment
    powered by a machine-learning model trained on thousands of real patterns.
  </p>
</div>
""", unsafe_allow_html=True)

# Model status ribbon
if model is None:
    st.error(
        "⚠️  **No trained model found.**  "
        "Run `python generate_data.py` then `python train.py` to generate the model.",
        icon="🚨",
    )
    st.stop()

col_a, col_b, col_c, col_d = st.columns(4)
all_metrics = meta.get("metrics", {})
col_a.metric("🏆 Active Model",   best_model_name)
col_b.metric("🎯 ROC-AUC",        f"{all_metrics.get('ROC-AUC', 0):.3f}")
col_c.metric("📐 F1 Score",        f"{all_metrics.get('F1', 0):.3f}")
col_d.metric("🔍 Recall",          f"{all_metrics.get('Recall', 0):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Customer Profile Inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👤 Customer Profile")
    st.caption("Fill in the details below, then click **Predict**.")
    st.markdown("---")

    st.markdown("#### 📋 Demographics & Tenure")
    age            = st.slider("Age", 18, 70, 35)
    tenure_months  = st.slider("Tenure (months)", 1, 72, 24)

    st.markdown("#### 💳 Billing")
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", 20.0, 9000.0,
                                       round(monthly_charges * tenure_months, 2), step=10.0)

    st.markdown("#### 🛒 Product & Support")
    num_products   = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    support_calls  = st.slider("Support Calls (last 12 mo.)", 0, 10, 2)
    satisfaction   = st.slider("Satisfaction Score (1 = worst)", 1, 5, 3)

    st.markdown("#### 📦 Service Details")
    contract_type    = st.selectbox("Contract Type",
                                    ["Month-to-Month", "One Year", "Two Year"])
    payment_method   = st.selectbox("Payment Method",
                                    ["Electronic Check", "Mailed Check",
                                     "Bank Transfer", "Credit Card"])
    internet_service = st.selectbox("Internet Service",
                                    ["DSL", "Fiber Optic", "No"])

    st.markdown("---")
    predict_btn = st.button("⚡ Predict Churn Risk", use_container_width=True)

    st.markdown("---")
    st.caption("Upload your own dataset to explore in the EDA tab.")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_eda, tab_compare, tab_help = st.tabs(
    ["🎯 Prediction", "📊 Data Explorer", "🏅 Model Comparison", "❓ Help & Explainability"]
)


# ── TAB 1: PREDICTION ─────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-header">Customer Risk Assessment</div>',
                unsafe_allow_html=True)

    if predict_btn:
        inputs = {
            "Age":               age,
            "Tenure_Months":     tenure_months,
            "Monthly_Charges":   monthly_charges,
            "Total_Charges":     total_charges,
            "Num_Products":      num_products,
            "Support_Calls":     support_calls,
            "Satisfaction_Score": satisfaction,
            "Contract_Type":     contract_type,
            "Payment_Method":    payment_method,
            "Internet_Service":  internet_service,
        }

        try:
            input_df = build_input_df(inputs, numeric_cols, categorical_cols)
            prob     = float(model.predict_proba(input_df)[0, 1])
            pred     = int(model.predict(input_df)[0])

            # ── Gauge chart ──────────────────────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Probability (%)", "font": {"size": 18}},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": "#ef4444" if pred else "#22c55e"},
                    "steps": [
                        {"range": [0,  35], "color": "#dcfce7"},
                        {"range": [35, 60], "color": "#fef9c3"},
                        {"range": [60, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 3},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c2:
                st.markdown(f"### {risk_label(prob)}")
                css_class = "churn-box" if pred else "safe-box"
                if pred:
                    verdict   = "⚠️ This customer is likely to churn."
                    action    = (
                        "Recommend a retention offer: consider a plan discount, "
                        "loyalty bonus, or a proactive support call. Act within "
                        "the next 30 days for the best impact."
                    )
                else:
                    verdict   = "✅ This customer is likely to stay."
                    action    = (
                        "Low risk detected. Continue delivering a great experience "
                        "and consider upsell opportunities for additional products."
                    )
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<div class="result-title">{verdict}</div>'
                    f'<div class="result-sub">{action}</div>'
                    f'</div>', unsafe_allow_html=True
                )

                st.markdown("#### 📌 Input Summary")
                summary_df = pd.DataFrame({
                    "Feature": list(inputs.keys()),
                    "Value":   list(inputs.values()),
                })
                st.dataframe(summary_df.set_index("Feature"), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Ensure you ran `python train.py` and the model file is present.")

    else:
        st.info(
            "👈  Fill in the **Customer Profile** in the sidebar, "
            "then click **Predict Churn Risk** to see the assessment here."
        )

        # Show sample insight cards
        if df_data is not None:
            st.markdown("#### 📈 Quick Dataset Stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers",    f"{len(df_data):,}")
            c2.metric("Churned",            f"{df_data['Churn'].sum():,}")
            c3.metric("Churn Rate",         f"{df_data['Churn'].mean():.1%}")
            c4.metric("Avg Monthly Charges",f"${df_data['Monthly_Charges'].mean():.2f}")


# ── TAB 2: EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # Resolve dataset (uploaded or default)
    try:
        if uploaded_file is not None:
            eda_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Using uploaded file — {len(eda_df):,} rows.")
        elif df_data is not None:
            eda_df = df_data.copy()
            st.caption("Using the default `customer_churn.csv` dataset.")
        else:
            st.warning("No dataset found. Upload a CSV or run `generate_data.py`.")
            eda_df = None
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        eda_df = None

    if eda_df is not None:
        # Row 1: Distribution + Churn by Contract
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("**Monthly Charges Distribution by Churn**")
            fig1 = px.histogram(
                eda_df, x="Monthly_Charges", color="Churn",
                barmode="overlay", nbins=40,
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                labels={"Churn": "Churned"},
                template="plotly_white",
            )
            fig1.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)

        with r1c2:
            st.markdown("**Churn Rate by Contract Type**")
            churn_by_contract = (
                eda_df.groupby("Contract_Type")["Churn"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Churn Rate", "count": "Customers"})
            )
            churn_by_contract["Churn Rate (%)"] = (churn_by_contract["Churn Rate"] * 100).round(1)
            fig2 = px.bar(
                churn_by_contract, x="Contract_Type", y="Churn Rate (%)",
                color="Churn Rate (%)", color_continuous_scale="Reds",
                text="Churn Rate (%)", template="plotly_white",
            )
            fig2.update_traces(texttemplate="%{text}%", textposition="outside")
            fig2.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Row 2: Tenure vs Monthly Charges scatter + Correlation heatmap
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**Tenure vs Monthly Charges**")
            sample = eda_df.sample(min(500, len(eda_df)), random_state=42)
            fig3 = px.scatter(
                sample, x="Tenure_Months", y="Monthly_Charges",
                color=sample["Churn"].map({0: "Stayed", 1: "Churned"}),
                color_discrete_map={"Stayed": "#22c55e", "Churned": "#ef4444"},
                opacity=0.65, template="plotly_white",
                labels={"color": "Status"},
            )
            fig3.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            st.markdown("**Feature Correlation Heatmap**")
            num_df  = eda_df.select_dtypes(include=[np.number])
            corr    = num_df.corr()
            fig4 = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, template="plotly_white",
                aspect="auto",
            )
            fig4.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

        # Row 3: Satisfaction score boxplot
        st.markdown("**Satisfaction Score vs Churn**")
        fig5 = px.box(
            eda_df, x=eda_df["Churn"].map({0: "Stayed", 1: "Churned"}),
            y="Satisfaction_Score",
            color=eda_df["Churn"].map({0: "Stayed", 1: "Churned"}),
            color_discrete_map={"Stayed": "#22c55e", "Churned": "#ef4444"},
            template="plotly_white", points="outliers",
            labels={"x": "Customer Status", "color": "Status"},
        )
        fig5.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

        with st.expander("🔍 Raw Data Preview"):
            st.dataframe(eda_df.head(100), use_container_width=True)


# ── TAB 3: MODEL COMPARISON ───────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-header">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    if meta and "all_metrics" in meta:
        df_compare = pd.DataFrame(meta["all_metrics"])
        df_compare = df_compare.set_index("Model")

        # Highlight best per column
        st.dataframe(
            df_compare.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                color="#bbf7d0",
            ).format("{:.4f}"),
            use_container_width=True,
        )

        # Radar chart
        st.markdown("**Head-to-Head Radar Chart**")
        metrics_radar = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        fig_radar = go.Figure()
        colors = ["#0ea5e9", "#ef4444", "#a855f7"]
        for (model_name, row), color in zip(df_compare.iterrows(), colors):
            vals = [row[m] for m in metrics_radar]
            vals += [vals[0]]  # close loop
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=metrics_radar + [metrics_radar[0]],
                fill="toself", name=model_name,
                line_color=color, fillcolor=color,
                opacity=0.25,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=420,
            template="plotly_white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.success(
            f"✅ **{best_model_name}** was selected as the active model "
            f"(highest ROC-AUC: **{df_compare.loc[best_model_name, 'ROC-AUC']:.4f}**)"
        )
    else:
        st.warning("Run `python train.py` to populate model comparison data.")


# ── TAB 4: HELP ───────────────────────────────────────────────────────────────
with tab_help:
    st.markdown('<div class="section-header">How ChurnGuard AI Works</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### 🧠 What is Churn Prediction?
    **Customer churn** means a customer stops using your service.
    This app uses a machine learning model to estimate the probability
    that a given customer will churn in the near future — *before* it happens —
    so you can take action.

    ---
    ### 📥 What Inputs Does the Model Use?

    | Feature | Why It Matters |
    |---|---|
    | **Age** | Older customers tend to have different loyalty patterns |
    | **Tenure (months)** | Longer customers are usually more loyal |
    | **Monthly Charges** | Higher bills increase churn risk |
    | **Total Charges** | Lifetime value proxy |
    | **Number of Products** | Multi-product customers are harder to churn |
    | **Support Calls** | Frequent complaints signal dissatisfaction |
    | **Satisfaction Score** | Direct self-reported sentiment |
    | **Contract Type** | Month-to-Month contracts churn far more often |
    | **Payment Method** | Electronic cheques correlate with higher churn |
    | **Internet Service** | Fibre users often have different expectations |

    ---
    ### ⚙️ How Does the Model Make a Decision?
    The model was trained on thousands of historical customer records.
    It learned which combinations of the above features predict whether a
    customer eventually left. When you enter a new customer's details,
    the model scores them on a **0 – 100% probability scale**.

    - **< 35%** → Low Risk (safe, continue normal engagement)
    - **35–60%** → Moderate Risk (monitor, consider soft retention offer)
    - **> 60%** → High Risk (proactive retention action recommended)

    ---
    ### 📏 Why ROC-AUC?
    We optimise for **ROC-AUC** rather than raw accuracy because churn datasets
    are *imbalanced* — most customers don't churn. AUC measures how well the
    model distinguishes churners from non-churners regardless of class imbalance.

    ---
    ### 🔒 Data & Privacy
    No data you enter in this app is stored or transmitted anywhere.
    All predictions happen locally inside your machine using the pre-trained model file.

    ---
    ### 🚀 Running the App Locally
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Generate the sample dataset
    python generate_data.py

    # 3. Train the models (takes ~30 seconds)
    python train.py

    # 4. Launch the app
    streamlit run app.py
    ```
    """)
