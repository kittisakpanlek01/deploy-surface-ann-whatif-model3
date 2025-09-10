import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

# =========================
# Load Model & Preprocessor
# =========================
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

num_cols = ["HNSPDI", "WNSPDI", "RMEXTG", "SLFUTI", "LSP_Body", "Entry_Body",
            "XVPTF8", "FT_HEAD", "CT_HEAD", "FTGM", "HDFBTH"]
cat_cols = ["QUASTR", "OPCCO", "LCBXON", "Product", "ENDUSE", "PASSNR"]

# =========================
# Helper Functions
# =========================
def preprocess_input(input_data):
    """Scale numerical + encode categorical features"""
    df = pd.DataFrame([input_data])

    # numeric
    X_num = scaler.transform(df[num_cols])

    # categorical
    X_cat = encoder.transform(df[cat_cols]).toarray()

    # combine
    X_all = np.concatenate([X_num, X_cat], axis=1)
    return X_all


def predict_with_softmax(X_all):
    """Predict probabilities and return P(No Defect)"""
    logits = model(X_all, training=False).numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs, logits


def compute_and_plot_shap(X_all):
    """Compute SHAP values for class 'No Defect' (index=2)"""
    try:
        background = X_all.astype(float)  # ใช้ input row เป็น background
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_all)

        # class index 2 = "No Defect"
        sv = shap_values.values[0, :, 2]
        base = shap_values.base_values[0, 2]

        feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))

        exp = shap.Explanation(
            values=sv,
            base_values=base,
            data=X_all[0],
            feature_names=feature_names
        )

        st.subheader("Local SHAP Explanation (Class: No Defect)")
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP computation failed: {e}")


# =========================
# Streamlit UI
# =========================
st.title("What-if Analysis: Steel Defect Prevention (ANN Model)")

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

if st.button("Run Prediction"):
    # preprocess
    X_all = preprocess_input(input_data)

    # predict
    probs, logits = predict_with_softmax(X_all)
    p_good = probs[0, 2]  # class index 2 = No Defect

    st.subheader("Prediction Results")
    st.write(f"**P(No Defect)** = {p_good:.2%}")
    st.write("All Probabilities (Softmax):", probs[0])

    # local SHAP explanation
    compute_and_plot_shap(X_all)
