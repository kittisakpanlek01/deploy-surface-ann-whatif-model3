# app_whatif.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import os

# ===========================
# โหลดโมเดลและ tools
# ===========================
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# กำหนด feature
num_cols = [
    "HNSPDI", "WNSPDI", "RMEXTG", "SLFUTI",
    "LSP_Body", "Entry_Body", "XVPTF8",
    "FT_HEAD", "CT_HEAD", "FTGM", "HDFBTH"
]
cat_cols = ["QUASTR", "OPCCO", "LCBXON", "Product", "ENDUSE", "PASSNR"]

# ===========================
# Streamlit UI
# ===========================
st.title("What-if Analysis: Surface Defect Prediction (ANN)")

st.sidebar.header("Input Parameters")

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
    "FTGM": st.sidebar.number_input("FM_FORCE", value=9000),
    "HDFBTH": st.sidebar.number_input("HDFBTH", value=18.0),
    "QUASTR": st.sidebar.selectbox("QUASTR", options=["C032RBB", "C032", "CG145", "CS0810", "CN1410", "CR1512"]),
    "OPCCO": st.sidebar.selectbox("OPCCO", options=["31", "0", "10", "21",  "41", "51", "66"]),
    "LCBXON": st.sidebar.selectbox("LCBXON", options=["USED CB", "BYPASS CB"]),
    "Product": st.sidebar.selectbox("PRODUCT", options=["PO/POx", "ColdRoll", "CutSheet", "Other",  "Stock"]),
    "ENDUSE": st.sidebar.selectbox("ENDUSE", options=["ADO", "PNX", "SDX", "FXX", "DGX", "ADH", "K1I", "GXX", "RST"]),
    "PASSNR": st.sidebar.selectbox("RM_PASS", options=["5", "7", "9"])
}

df_input = pd.DataFrame([input_data])

# ===========================
# Prediction
# ===========================
if st.button("Run Prediction"):
    try:
        # --- preprocess ---
        X_num = df_input[num_cols]
        X_cat = df_input[cat_cols]

        X_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_all = np.hstack((X_scaled, X_cat_encoded)).reshape(1, -1)

        # --- predict ---
        pred = model.predict(X_all)
        probs = tf.nn.softmax(pred).numpy()[0]

        # P(No Defect) = index 3
        p_good = probs[2] if len(probs) > 3 else None
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {pred_label}")
        if p_good is not None:
            st.write(f"**P(No Defect): {p_good:.2%}**")
        else:
            st.warning("⚠ Model ไม่มี class index 3 → ตรวจสอบ label encoder")

        # ===========================
        # Local SHAP Explanation
        # ===========================
        try:
            st.subheader("Local SHAP Explanation")

            # ใช้ background (sampling เล็ก ๆ จะเสถียรกว่า)
            background = X_all
            explainer = shap.Explainer(model, background)
            shap_values = explainer(X_all)  # ออกมาเป็น multi-output

            # เลือก class "No Defect" = index 3
            if shap_values.values.ndim == 3:
                sv = shap_values.values[0, :, 3]   # sample 0, features :, class index 3
            elif shap_values.values.ndim == 2:
                sv = shap_values.values[0]         # กรณี binary หรือ single output
            else:
                raise ValueError("Unexpected SHAP output shape")

            # plot waterfall
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap.Explanation(
                values=sv,
                base_values=shap_values.base_values[0, 3] if shap_values.values.ndim == 3 else shap_values.base_values[0],
                data=X_all[0],
                feature_names=num_cols + list(encoder.get_feature_names_out(cat_cols))
            ), show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP computation failed: {e}")

        # ===========================
        # Suggestion Engine (simple)
        # ===========================
        st.subheader("Suggestion Engine (Δ adjustment)")
        suggestions = []
        if p_good is not None and p_good < 0.8:
            if input_data["CT_HEAD"] < 600:
                suggestions.append("↑ เพิ่ม CT_HEAD ประมาณ +20")
            if input_data["FT_HEAD"] < 900:
                suggestions.append("↑ เพิ่ม FT_HEAD ประมาณ +15")
            if input_data["XVPTF8"] < 10:
                suggestions.append("↑ SPEED ขึ้นเล็กน้อย (+1 ~ +2)")
            if input_data["FTGM"] > 10000:
                suggestions.append("↓ ลด FM_FORCE ประมาณ -500")

        if suggestions:
            for s in suggestions:
                st.write("- ", s)
        else:
            st.write("✅ ไม่มีคำแนะนำเพิ่มเติม (ค่าปัจจุบันเหมาะสม)")

        # ===========================
        # Log Monitoring
        # ===========================
        log_entry = {
            "input": input_data,
            "pred_label": pred_label,
            "p_good": float(p_good) if p_good is not None else None,
            "suggestions": suggestions
        }

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/policy_log.csv"

        if not os.path.exists(log_file):
            pd.DataFrame([log_entry]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv(log_file, mode="a", header=False, index=False)

        st.info("บันทึก log → policy_log.csv")

    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")
