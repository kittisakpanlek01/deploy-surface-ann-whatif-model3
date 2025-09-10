import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# Load model and preprocessing
# -------------------------------
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

num_cols = ['HNSPDI', 'WNSPDI', 'RMEXTG', 'SLFUTI', 'LSP_Body',
            'Entry_Body', 'XVPTF8', 'FT_HEAD', 'CT_HEAD', 'FTGM', 'HDFBTH']
cat_cols = ['QUASTR', 'OPCCO', 'LCBXON', 'Product', 'ENDUSE', 'PASSNR']

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Surface Defect Prediction (ANN) with What-if Analysis")

# Default inputs
input_data = {
    "HNSPDI": st.number_input("THICKNESS", value=4.0),
    "WNSPDI": st.number_input("WIDTH", value=1219.0),
    "RMEXTG": st.number_input("BAR_THICK", value=38.0),
    "SLFUTI": st.number_input("TIME_IN_FUR", value=3.5),
    "LSP_Body": st.slider("LSP_Body", 1000.0, 1200.0, 1100.0, step=5.0),
    "Entry_Body": st.slider("Entry_Body", 1000.0, 1200.0, 1040.0, step=5.0),
    "XVPTF8": st.slider("SPEED", 1.0, 20.0, 8.0, step=0.5),
    "FT_HEAD": st.slider("FT_HEAD", 500.0, 1200.0, 860.0, step=5.0),
    "CT_HEAD": st.slider("CT_HEAD", 400.0, 800.0, 540.0, step=5.0),
    "FTGM": st.slider("FM_FORCE", 1000.0, 15000.0, 9000.0, step=100.0),
    "HDFBTH": st.slider("HDFBTH", 5.0, 30.0, 18.0, step=0.5),
    "QUASTR": st.selectbox("QUASTR", ["C032RBB", "C032",  "CG145", "CS0810", "CN1410", "CR1512"]),
    "OPCCO": st.selectbox("OPCCO", ["31", "0", "10", "21", "41", "51", "66"]),
    "LCBXON": st.selectbox("LCBXON", ["USED CB", "BYPASS CB"]),
    "Product": st.selectbox("PRODUCT", ["PO/POx", "ColdRoll", "CutSheet", "Other",  "Stock"]),
    "ENDUSE": st.selectbox("ENDUSE", ["ADO", "PNX", "SDX", "FXX", "DGX", "ADH", "K1I", "GXX", "RST"]),
    "PASSNR": st.selectbox("RM_PASS", ["5", "7", "9"])
}

df_input = pd.DataFrame([input_data])

if st.button("Predict & Explain"):
    try:
        # -------------------------------
        # Preprocess
        # -------------------------------
        X_num = df_input[num_cols]
        X_cat = df_input[cat_cols]

        X_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()

        X_all = np.hstack((X_scaled, X_cat_encoded)).reshape(1, -1)

        # -------------------------------
        # Prediction
        # -------------------------------
        preds = model.predict(X_all)
        probs = tf.nn.softmax(preds).numpy()[0]

        p_no_defect = probs[2]   # class index 2 = "No Defect"
        pred_label = label_encoder.inverse_transform([np.argmax(probs)])

        st.metric("Prediction", pred_label[0])
        st.metric("P(No Defect)", f"{p_no_defect:.2%}")

        # -------------------------------
        # Local SHAP Explanation
        # -------------------------------
        background = X_all
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_all)

        # ดึงค่า shap เฉพาะ class 2 = No Defect
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

        # -------------------------------
        # Simple Suggestion Engine
        # -------------------------------
        st.subheader("Suggestions to Improve P(No Defect)")
        top_bad = sorted(zip(feature_names, sv, X_all[0]), key=lambda x: x[1])[:3]
        for feat, impact, val in top_bad:
            if impact < 0:
                st.write(f"- Try adjusting **{feat}** (current={val:.2f}), negative impact {impact:.3f}")

    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")
