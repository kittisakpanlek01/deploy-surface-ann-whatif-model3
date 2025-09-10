import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --------------------------
# โหลดโมเดลและ Tools
# --------------------------
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------------------------
# Feature Groups
# --------------------------
num_cols = ['HNSPDI','WNSPDI','RMEXTG','SLFUTI','LSP_Body','Entry_Body',
            'XVPTF8','FT_HEAD','CT_HEAD','FTGM','HDFBTH']
cat_cols = ['QUASTR','OPCCO','LCBXON','Product','ENDUSE','PASSNR']

# --------------------------
# Suggest Changes Function
# --------------------------
def suggest_changes(df_input, base_prob, step=0.05):
    """จำลองการปรับ feature ทีละนิด แล้วบอกว่า Δ เท่าไรที่ช่วยเพิ่ม P(Good)"""
    suggestions = []
    for col in num_cols:
        test_df = df_input.copy()
        test_df[col] = test_df[col] * (1 + step)  # เพิ่มขึ้นเล็กน้อย
        X_num = scaler.transform(test_df[num_cols])
        X_cat = encoder.transform(test_df[cat_cols]).toarray()
        X_all = np.hstack((X_num, X_cat))
        preds = model.predict(X_all, verbose=0)
        prob_good = preds[0][np.argmax(preds)]
        if prob_good > base_prob:
            suggestions.append({
                "feature": col,
                "change": f"+{step*100:.1f}%",
                "delta_prob": prob_good - base_prob
            })
    return suggestions

# --------------------------
# SHAP Computation Function
# --------------------------
def compute_shap_values(X_all, preds):
    """
    คำนวณ SHAP values สำหรับ class 'Good' (index 0)
    """
    try:
        explainer = shap.DeepExplainer(model, X_all)
        shap_values = explainer.shap_values(X_all)
    except Exception:
        background = X_all + np.random.normal(0, 0.01, X_all.shape)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_all, nsamples=50)

    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))

    # ✅ Fix: อธิบายเฉพาะ class 0 (Good)
    sv = shap_values[2][0]   # class 0, sample 0

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": sv,
        "Value": X_all[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    fig = px.bar(
        shap_df.head(10),
        x="SHAP Value", y="Feature",
        orientation="h",
        title=f"Top SHAP Feature Impacts (Class: Good)"
    )
    return shap_df, fig

# --------------------------
# Logging Function
# --------------------------
def append_log(df_input, pred_label, suggestions, logfile="policy_log.csv"):
    """บันทึก Input + Prediction + Suggestions ลงไฟล์ CSV"""
    log_entry = df_input.copy()
    log_entry["Prediction"] = pred_label[0]
    log_entry["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry["Suggestions"] = str(suggestions)

    if os.path.exists(logfile):
        log_df = pd.read_csv(logfile)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
    else:
        log_df = log_entry

    log_df.to_csv(logfile, index=False)

# --------------------------
# UI Layout
# --------------------------
st.set_page_config(page_title="What-if Surface Defect", layout="wide")
st.title("🔎 What-if Analysis: Surface Defect Prevention")

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
    "FTGM": st.sidebar.number_input("FM_FORCE", value=9000.0),
    "HDFBTH": st.sidebar.number_input("HDFBTH", value=18.0),
    "QUASTR": st.sidebar.selectbox("QUASTR", ["C032","C032RBB","CG145","CS0810","CN1410","CR1512"]),
    "OPCCO": st.sidebar.selectbox("OPCCO", ["0","10","21","31","41","51","66"]),
    "LCBXON": st.sidebar.selectbox("LCBXON", ["USED CB","BYPASS CB"]),
    "Product": st.sidebar.selectbox("PRODUCT", ["ColdRoll","CutSheet","Other","PO/POx","Stock"]),
    "ENDUSE": st.sidebar.selectbox("ENDUSE", ["PNX","SDX","FXX","DGX","ADO","ADH","K1I","GXX","RST"]),
    "PASSNR": st.sidebar.selectbox("RM_PASS", ["5","7","9"])
}

df_input = pd.DataFrame([input_data])

# --------------------------
# Prediction + SHAP
# --------------------------
if st.button("🔮 Predict + Explain"):
    try:
        # เตรียมข้อมูล
        X_num = df_input[num_cols]
        X_cat = df_input[cat_cols]
        X_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_all = np.hstack((X_scaled, X_cat_encoded))

        # พยากรณ์
        preds = model.predict(X_all, verbose=0)
        pred_label = label_encoder.inverse_transform([np.argmax(preds)])
        base_prob = preds[0][np.argmax(preds)]

        st.subheader("📌 Prediction Result")
        st.success(f"Prediction: {pred_label[0]}")
        st.write("Probability (per class):", preds.tolist())

        # Suggestion Engine
        st.subheader("🛠 Suggested Feature Adjustments")
        suggestions = suggest_changes(df_input, base_prob)
        if suggestions:
            st.write(pd.DataFrame(suggestions))
        else:
            st.info("No simple improvement found with +5% simulation.")

        # SHAP Local Explanation
        st.subheader("📊 Local SHAP Explanation")
        shap_df, fig = compute_shap_values(X_all, preds)
        st.dataframe(shap_df.head(10))
        st.pyplot(fig)

        # Logging
        append_log(df_input, pred_label, suggestions)

    except Exception as e:
        st.error(f"Prediction/SHAP error: {str(e)}")


# ---------------------------
# ส่วน prediction ใน Streamlit
# ---------------------------
if st.button("Predict"):
    try:
        preds = model.predict(input_data_scaled)
        prob_good = float(preds[0][0])   # ✅ index 0 = Good
        st.metric("P(Good)", f"{prob_good:.2%}")

        shap_df, fig = compute_shap_values(input_data_scaled, preds)
        st.subheader("Local SHAP Explanation (Why this prediction?)")
        st.dataframe(shap_df.head(10))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")

