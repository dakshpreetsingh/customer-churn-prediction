import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import add_engineered_features

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

st.set_page_config(page_title="Customer Churn Predictor",
                   page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.joblib")
    with open("model_config.json") as f:
        cfg = json.load(f)
    prep = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    explainer = None
    if SHAP_OK:
        try:
            if cfg["model_name"] in ("XGBoost", "Random Forest"):
                explainer = shap.TreeExplainer(clf)
            else:
                explainer = shap.LinearExplainer(clf, np.load("shap_background.npy"))
        except Exception:
            explainer = None
    return model, prep, cfg, explainer


model, prep, cfg, explainer = load_artifacts()
THRESHOLD = cfg["threshold"]
AVG_CHARGE = cfg["avg_monthly_charge"]
YES_NO = {"Yes": 1, "No": 0}

st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details in the sidebar. The model estimates "
           "churn risk and shows which factors drive it.")

# ───────────────────── SIDEBAR INPUTS ─────────────────────
with st.sidebar:
    st.header("Customer details")

    with st.expander("Account", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        monthly = st.number_input("Monthly charges", min_value=0.0,
                                  max_value=200.0, value=85.0, step=1.0)
        auto_total = st.checkbox("Estimate total charges (monthly × tenure)", value=True)
        if auto_total:
            total = monthly * tenure
        else:
            total = st.number_input("Total charges", min_value=0.0,
                                    value=float(monthly * tenure), step=10.0)

    with st.expander("Services"):
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = (st.selectbox("Multiple lines", ["No", "Yes"])
                    if phone == "Yes" else "No phone service")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

        addon_labels = {
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
        }
        addons = {}
        for col, label in addon_labels.items():
            if internet == "No":
                addons[col] = "No internet service"
            else:
                addons[col] = st.selectbox(label, ["No", "Yes"])

    with st.expander("Demographics"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])

# ───────────────────── BUILD THE MODEL INPUT ─────────────────────
raw = pd.DataFrame([{
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": YES_NO[senior],
    "Partner": YES_NO[partner],
    "Dependents": YES_NO[dependents],
    "tenure": tenure,
    "PhoneService": YES_NO[phone],
    "MultipleLines": multiple,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": YES_NO[paperless],
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}])

X_row = add_engineered_features(raw, AVG_CHARGE)[cfg["feature_columns"]]
score = float(model.predict_proba(X_row)[0, 1])
flagged = score >= THRESHOLD

# ───────────────────── RESULTS ─────────────────────
left, right = st.columns([1, 1.5])

with left:
    st.subheader("Prediction")
    st.metric("Churn risk score", f"{score:.0%}")
    st.progress(min(max(score, 0.0), 1.0))
    if flagged:
        st.error(f"⚠️ High risk: above the model's decision threshold ({THRESHOLD:.0%}).")
    else:
        st.success(f"✅ Lower risk: below the model's decision threshold ({THRESHOLD:.0%}).")
    st.caption("This is a risk score, not a calibrated probability. The model was "
               "trained on resampled (SMOTE) data, so scores run higher than the "
               "real-world churn rate. Use it to rank customers.")

with right:
    st.subheader("Why this score?")
    if explainer is None:
        st.info("Explanations are unavailable in this environment.")
    else:
        try:
            Xt = prep.transform(X_row)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                vals = sv[1][0]
            elif np.ndim(sv) == 3:
                vals = sv[0, :, 1]
            else:
                vals = sv[0]

            contrib = pd.Series(vals, index=cfg["transformed_feature_names"])
            top = contrib.loc[contrib.abs().sort_values(ascending=False).head(8).index]

            fig, ax = plt.subplots(figsize=(6, 3.6))
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in top.values[::-1]]
            ax.barh(list(top.index[::-1]), list(top.values[::-1]), color=colors)
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("Effect on risk score  (left lowers, right raises)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            ups = [n for n, v in contrib.sort_values(ascending=False).head(3).items() if v > 0]
            downs = [n for n, v in contrib.sort_values().head(2).items() if v < 0]
            if ups:
                st.markdown("**Raising risk:** " + ", ".join(f"`{n}`" for n in ups))
            if downs:
                st.markdown("**Lowering risk:** " + ", ".join(f"`{n}`" for n in downs))
        except Exception as e:
            st.warning(f"Could not compute the explanation: {e}")

# ───────────────────── WHAT-IF ─────────────────────
st.divider()
st.subheader("What-if: contract type")
rows = []
for c in ["Month-to-month", "One year", "Two year"]:
    alt = raw.copy()
    alt["Contract"] = c
    X_alt = add_engineered_features(alt, AVG_CHARGE)[cfg["feature_columns"]]
    rows.append({"Contract": c,
                 "Risk score": float(model.predict_proba(X_alt)[0, 1])})
st.bar_chart(pd.DataFrame(rows).set_index("Contract"))
st.caption("How the model's score changes if only the contract differs. This shows "
           "model sensitivity, not a guaranteed causal effect; confirming it would "
           "need an A/B test.")

st.caption(f"Model: {cfg['model_name']} · decision threshold {THRESHOLD:.2f} · "
           "trained on the Telco Customer Churn dataset")
