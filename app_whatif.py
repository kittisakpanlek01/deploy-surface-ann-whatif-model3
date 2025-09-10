import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
# import plotly.express as px
import matplotlib.pyplot as plt
import os

# --------------------------
# Load model and preprocessors
# --------------------------
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Columns
num_cols = ['HNSPDI', 'WNSPDI', 'RMEXTG', 'SLFUTI',
            'LSP_Body', 'Entry_Body', 'XVPTF8', 'FT_HEAD',
            'CT_HEAD', 'FTGM', 'HDFBTH']
cat_cols = ['QUASTR', 'OPCCO', 'LCBXON',
            'Product', 'ENDUSE', 'PASSNR']

# --------------------------
# Helper functions
# --------------------------
def preprocess_input(df_input):
    """Scale numeric + encode categorical"""
    X_num = scaler.transform(df_input[num_cols])
    X_cat = encoder.transform(df_input[cat_cols]).toarray()
    return np.hstack((X_num, X_cat))


def compute_shap_values(X_all):
    """คำนวณ SHAP values สำหรับ class 'No Defect' (index=3)"""
    try:
        explainer = shap.DeepExplainer(model, X_all)
        shap_values = explainer.shap_values(X_all)
    except Exception:
        background = X_all + np.random.normal(0, 0.01, X_all.shape)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_all, nsamples=50)

    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))

    sv = shap_values[3][0]  # ✅ class index 3 = "No Defect"

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": sv,
        "Value": X_all[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    # fig = px.bar(
    #     shap_df.head(10),
    #     x="SHAP Value", y="Feature",
    #     orientation="h",
    #     title=f"Top SHAP Feature Impacts (Class: No Defect)"
    # )
    fig, ax = plt.subplots(figsize=(8,5))
    shap_df.head(10).plot(
        kind="barh", x="Feature", y="SHAP Value", ax=ax, legend=False
    )
    plt.tight_layout()
    return shap_df, fig


def suggest_changes(X_all, feature, step=0.05):
    """Simple simulation: ลองปรับ feature แล้วดูว่า P(No Defect) ดีขึ้นไหม"""
    base_pred = model.predict(X_all)[0][3]  # P(No Defect)
    suggestions = []
    for delta in [-2, -1, 1, 2]:
        X_temp = X_all.copy()
        col_idx = num_cols.index(feature) if feature in num_cols else None
        if col_idx is not None:
            X_temp[0, col_idx] += delta * step
            new_pred = model.predict(X_temp)[0][3]
            suggestions.append((delta, new_pred))
    best = max(suggestions, key=lambda x: x[1])
    return base_pred, best


def append_log(input_dict, prob_good, suggestions):
    """บันทึกคำแนะนำ → การปรับจริง → ผลลัพธ์"""
    log_file = "whatif_log.csv"
    new_row = {"input": str(input_dict),
               "P_NoDefect": prob_good,
               "suggestions": str(suggestions)}

    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        df_log = pd.concat([df_log, pd.DataFrame([new_row])],
                           ignore_index=True)
    else:
        df_log = pd.DataFrame([new_row])

    df_log.to_csv(log_file, index=False)


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="What-if: Surface Defect", layout="wide")
st.title("🔎 What-if Analysis: Surface Defect Prediction (ANN)")

st.sidebar.header("Input Features")
input_data = {
    "HNSPDI": st.sidebar.number_input("THICKNESS", value=4.0),
    "WNSPDI": st.sidebar.number_input("WIDTH", value=1219.0),
    "RMEXTG": st.sidebar.number_input("BAR_THICK", value=38.0),
    "SLFUTI": st.sidebar.number_input("TIME_IN_FUR", value=3.5),
    "LSP_Body": st.sidebar.number_input("LSP_Body", value=1100.0),
    "Entry_Body": st.sidebar.number_input("Entry_Body", value=1040.0),
    "XVPTF8": st.sidebar.number_input("SPEED", value=8.0),
    "FT_HEAD": st.sidebar.number_input("FT_HEAD", value=860.0),
    "CT_HEAD": st.sidebar.number_input("CT_HEAD", value=540.0),
    "FTGM": st.sidebar.number_input("FM_FORCE", value=9000.0),
    "HDFBTH": st.sidebar.number_input("HDFBTH", value=18.0),
    "QUASTR": st.sidebar.selectbox("QUASTR", options=["C032", "C032RBB", "CG145", "CS0810", "CN1410", "CR1512"]),
    "OPCCO": st.sidebar.selectbox("OPCCO", options=["0", "10", "21", "31", "41", "51", "66"]),
    "LCBXON": st.sidebar.selectbox("LCBXON", options=["USED CB", "BYPASS CB"]),
    "Product": st.sidebar.selectbox("PRODUCT", options=["ColdRoll", "CutSheet", "Other", "PO/POx", "Stock"]),
    "ENDUSE": st.sidebar.selectbox("ENDUSE", options=["PNX", "SDX", "FXX", "DGX", "ADO", "ADH", "K1I", "GXX", "RST"]),
    "PASSNR": st.sidebar.selectbox("RM_PASS", options=["5", "7", "9"])
}
df_input = pd.DataFrame([input_data])
X_all = preprocess_input(df_input)

if st.button("🔮 Predict"):
    try:
        preds = model.predict(X_all)
        prob_good = float(preds[0][3])  # ✅ Index 3 = No Defect
        st.metric("P(No Defect)", f"{prob_good:.2%}")

        shap_df, fig = compute_shap_values(X_all)
        st.subheader("Local SHAP Explanation (Why this prediction?)")
        st.dataframe(shap_df.head(10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Suggestions")
        suggestions = {}
        for f in ["CT_HEAD", "FT_HEAD", "XVPTF8", "FTGM", "HDFBTH", "LSP_Body", "Entry_Body"]:
            base_pred, best = suggest_changes(X_all, f, step=0.1)
            suggestions[f] = {"Δ": best[0], "New_P": best[1]}
            st.write(f"- {f}: Δ{best[0]} → P(No Defect) = {best[1]:.2%}")

        append_log(input_data, prob_good, suggestions)

    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")

