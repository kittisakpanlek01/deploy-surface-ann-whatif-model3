# app_whatif.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import datetime
import os
import json
import re
import shap
import matplotlib.pyplot as plt

# --- โหลด model & preprocessors (ปรับ path ตามจริง) ---
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- โหลดข้อมูล background สำหรับ SHAP (สำคัญมาก) ---
# SHAP KernelExplainer ทำงานได้ดีที่สุดเมื่อมีข้อมูลตัวอย่างเพื่อใช้เป็น "พื้นหลัง"
try:
    X_train_sample = pd.read_csv("X_train_sample_processed.csv", header=None).values
    st.sidebar.success("Loaded SHAP background data.")
# Around line 25
except FileNotFoundError:
    st.sidebar.error("ไม่พบไฟล์ 'X_train_sample_processed.csv' สำหรับ SHAP")
    st.sidebar.warning("SHAP explanation อาจทำงานไม่ถูกต้อง (ใช้ข้อมูลจำลอง)")
    # สร้างข้อมูลจำลองขึ้นมาแทนชั่วคราว ถ้าไม่พบไฟล์ เพื่อไม่ให้แอปพัง
    if 'model' in locals() and hasattr(model, 'input_shape'):
        # Create 100 dummy samples instead of 1
        X_train_sample = np.zeros((100, model.input_shape[1])) # <-- FIX
    else:
        # ใส่จำนวน features โดยประมาณ หาก model ยังไม่ถูกโหลด
        X_train_sample = np.zeros((100, 50)) # <-- FIX


# features (ปรับให้ตรงกับของคุณ)
num_cols = ['HNSPDI','WNSPDI','RMEXTG','SLFUTI','LSP_Body','Entry_Body','XVPTF8','FT_HEAD','CT_HEAD','FTGM','HDFBTH']
cat_cols = ['QUASTR','OPCCO','LCBXON','Product','ENDUSE','PASSNR']

# ตัวแปรที่อนุญาตให้ปรับได้ (actionable)
actionable = {
    "CT_HEAD": {"min": 400, "max": 800, "step": 10},
    "FT_HEAD": {"min": 700, "max": 950, "step": 10},
    "RMEXTG": {"min": 26, "max": 40, "step": 2},
    "SLFUTI": {"min": 2, "max": 4, "step": 0.2},
    "XVPTF8": {"min": 2.0, "max": 14.0, "step": 0.2},
    "FTGM": {"min": 1000, "max": 16000, "step": 200},
    "HDFBTH": {"min": 1, "max": 28, "step": 1},
    "LSP_Body": {"min": 1000, "max": 1200, "step": 20},
    "Entry_Body": {"min": 800, "max": 1100, "step": 20},
}

LOGFILE = "policy_log.csv"

# --- helper: prepare X from a dict of features ---
def build_X_from_dict(features_dict):
    """Return X_all ready for model.predict (numpy array) given a dict with numeric+categorical."""
    df_num = pd.DataFrame([{c: features_dict.get(c, np.nan) for c in num_cols}])
    df_cat = pd.DataFrame([{c: features_dict.get(c, "") for c in cat_cols}])
    X_scaled = scaler.transform(df_num)
    X_cat_encoded = encoder.transform(df_cat).toarray()
    X_all = np.hstack((X_scaled, X_cat_encoded))
    return X_all, df_num, df_cat

# --- helper: get predicted probabilities + predicted label ---
def predict_probs(features_dict):
    X_all, _, _ = build_X_from_dict(features_dict)
    probs = model.predict(X_all).flatten()
    classes = label_encoder.classes_
    prob_series = pd.Series(probs, index=classes)
    pred_class = classes[np.argmax(probs)]
    return prob_series, pred_class

# --- helper: find index for "Good/No Defect" class if possible ---
def find_good_label_index():
    classes = list(label_encoder.classes_)
    for i, cls in enumerate(classes):
        if re.search(r"no[\s_-]*defect|no[\s_-]*def|good|ok|pass|non-defect", str(cls), re.I):
            return i
    return None

GOOD_IDX = find_good_label_index()

# --- suggestion engine (simple simulation) ---
def suggest_changes(base_features, target_improve=0.05, n_steps=11, max_delta_fraction=0.1):
    base_prob_series, base_pred = predict_probs(base_features)
    if GOOD_IDX is not None:
        base_pgood = base_prob_series.iloc[GOOD_IDX]
    else:
        base_pgood = base_prob_series[base_pred]
    suggestions = []
    for feat, cfg in actionable.items():
        cur = float(base_features.get(feat, np.nan))
        if np.isnan(cur): continue
        span = max(1.0, abs(cur) * max_delta_fraction)
        candidates = np.clip(np.linspace(cur - span, cur + span, n_steps), cfg["min"], cfg["max"])
        best = {"feat": feat, "cur": cur, "best_val": cur, "best_pgood": base_pgood, "delta": 0.0}
        for val in candidates:
            trial = base_features.copy()
            trial[feat] = float(val)
            prob_series, _ = predict_probs(trial)
            pgood = prob_series.iloc[GOOD_IDX] if GOOD_IDX is not None else prob_series.max()
            if pgood > best["best_pgood"]:
                best.update({"best_val": val, "best_pgood": float(pgood), "delta": float(val-cur)})
        improvement = best["best_pgood"] - base_pgood
        unit = abs(best["delta"]) if abs(best["delta"]) > 1e-9 else 1e-9
        gain_per_unit = improvement / unit
        suggestions.append({**best, "improvement": float(improvement), "gain_per_unit": float(gain_per_unit)})
    return base_prob_series, base_pgood, sorted(suggestions, key=lambda x: (x["improvement"], x["gain_per_unit"]), reverse=True)

# --- Logging function ---
def append_log(entry: dict):
    df_entry = pd.DataFrame([entry])
    if not os.path.exists(LOGFILE):
        df_entry.to_csv(LOGFILE, index=False)
    else:
        df_entry.to_csv(LOGFILE, mode="a", header=False, index=False)


# # --- SHAP explanation (REVISED) ---
# def compute_and_plot_shap(instance_dict, background_data):
#     """
#     คำนวณ SHAP values และสร้าง waterfall plot โดยใช้ background data ที่เหมาะสม
#     """
#     try:
#         # 1. เตรียมข้อมูล instance ที่จะอธิบาย
#         X_instance, _, _ = build_X_from_dict(instance_dict)

#         # 2. สร้าง Explainer ด้วย background data
#         # ใช้ K-Means เพื่อสรุป background data ให้มีขนาดเล็กลง (เร็วขึ้น)
#         background_summary = shap.kmeans(background_data, 15)
#         explainer = shap.KernelExplainer(model.predict, background_summary)

#         # 3. คำนวณ SHAP values สำหรับ instance ของเรา
#         shap_values = explainer.shap_values(X_instance)

#         # 4. หา Index ของ Class 'Good' หรือ 'No Defect'
#         good_class_idx = find_good_label_index()
#         if good_class_idx is None:
#             probs, _ = predict_probs(instance_dict)
#             good_class_idx = np.argmax(probs.values)
#             st.warning(f"ไม่พบ Class 'Good/No Defect', จึงใช้ Class ที่มีโอกาสสูงสุดแทน: '{label_encoder.classes_[good_class_idx]}'")

#         # 5. สร้าง SHAP Explanation Object
#         explanation = shap.Explanation(
#             values=shap_values[good_class_idx][0],
#             base_values=explainer.expected_value[good_class_idx],
#             data=X_instance[0],
#             feature_names=num_cols + list(encoder.get_feature_names_out(cat_cols))
#         )

#         # 6. สร้าง Plot
#         fig, ax = plt.subplots()
#         shap.plots.waterfall(explanation, max_display=15, show=False)
#         fig.tight_layout()
#         return fig, None

#     except Exception as e:
#         import traceback
#         st.text(traceback.format_exc()) # แสดง error แบบละเอียดเพื่อ debug
#         return None, f"SHAP computation failed: {e}"

def compute_and_plot_shap(df_input):
    try:
        X_num = df_input[num_cols]
        X_cat = df_input[cat_cols]
        X_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_all = np.hstack((X_scaled, X_cat_encoded))

        background_data = X_all[np.random.choice(X_all.shape[0], min(50, X_all.shape[0]), replace=False)]
        background_summary = shap.kmeans(background_data, 15)

        explainer = shap.KernelExplainer(model.predict, background_summary)
        shap_values = explainer.shap_values(X_all, nsamples=50)

        target_class = 2  # No Defect
        shap_value_single = shap_values[target_class][0]

        fig, ax = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_value_single,
                base_values=explainer.expected_value[target_class],
                data=X_all[0],
                feature_names=num_cols + list(encoder.get_feature_names_out(cat_cols))
            ),
            show=False
        )
        return fig, None
    except Exception as e:
        return None, str(e)



# ----------------- Streamlit UI -----------------
st.title("What-If: Human-in-the-loop Recommendations")

st.write("ปรับค่า setpoints แบบทดลอง (What-If) และรับคำแนะนำ Δ ที่เป็นไปได้ พร้อม SHAP explanation และบันทึกการตัดสินใจของผู้ปฏิบัติงาน")

st.subheader("1) กรอกค่าปัจจุบันของคอยล์ (Current features)")
base_features = {}
col1, col2, col3 = st.columns(3)
with col1:
    for c in ["HNSPDI","WNSPDI","RMEXTG","SLFUTI"]:
        default = 4.0 if c=="HNSPDI" else (1219.0 if c=="WNSPDI" else (38.0 if c=="RMEXTG" else 3.5))
        base_features[c] = st.number_input(c, value=float(default))
with col2:
    for c in ["CT_HEAD","FT_HEAD","XVPTF8","FTGM"]:
        default = 540 if c=="CT_HEAD" else (860 if c=="FT_HEAD" else (8.0 if c=="XVPTF8" else 9000))
        base_features[c] = st.number_input(c, value=float(default))
with col3:
    for c in ["HDFBTH","LSP_Body","Entry_Body"]:
        default = 18.0 if c=="HDFBTH" else (1110.0 if c=="LSP_Body" else 1040.0)
        base_features[c] = st.number_input(c, value=float(default))

# --- ส่วนที่แก้ไข: ใช้ st.selectbox สำหรับ Categorical Features ---
st.markdown("**Categorical features**")
cat_defaults = {}
col1_cat, col2_cat = st.columns(2)

with col1_cat:
    cat_defaults["QUASTR"] = st.selectbox("QUASTR", ["C032RBB", "C032", "CG145", "CS0810", "CN1410", "CR1512"])
    cat_defaults["OPCCO"] = st.selectbox("OPCCO", ["31", "0", "10", "21", "41", "51", "66"])
    cat_defaults["LCBXON"] = st.selectbox("LCBXON", ["USED CB", "BYPASS CB"])
with col2_cat:
    cat_defaults["Product"] = st.selectbox("Product", ["PO/POx", "ColdRoll", "CutSheet", "Other", "Stock"])
    cat_defaults["ENDUSE"] = st.selectbox("ENDUSE", ["ADO", "PNX", "SDX", "FXX", "DGX", "ADH", "K1I", "GXX", "RST"])
    cat_defaults["PASSNR"] = st.selectbox("PASSNR", ["5", "7", "9"])
# ----------------------------------------------------------------

if st.button("Predict current P(classes)"):
    try:
        # รวม dict ของ features ทั้งหมดก่อนส่งไปทำนาย
        current_features = {**base_features, **cat_defaults}
        probs, pred = predict_probs(current_features)
        
        st.write("Probabilities per class:")
        st.dataframe(probs.to_frame("probability"))
        st.write("Predicted class:", pred)
        if GOOD_IDX is not None:
            st.info(f"P(good) estimated (class '{label_encoder.classes_[GOOD_IDX]}') = {probs.iloc[GOOD_IDX]:.4f}")
    except Exception as e:
        st.error(f"Predict error: {e}")

st.write("---")
st.subheader("2) ข้อเสนอแนะ (Suggest changes)")
target_improve = st.slider("เป้าหมายเพิ่ม P(Good) ขั้นต่ำ (absolute)", 0.0, 0.5, 0.05, 0.01)
if st.button("Generate suggestions"):
    with st.spinner("กำลังคำนวณข้อเสนอแนะ..."):
        try:
            current_features = {**base_features, **cat_defaults}
            base_prob_series, base_pgood, suggestions = suggest_changes(current_features, target_improve=target_improve)
            
            st.write("Base probabilities:")
            st.dataframe(base_prob_series.to_frame("prob"))
            if GOOD_IDX is not None:
                st.write(f"Base P(good) = {base_pgood:.4f} (class '{label_encoder.classes_[GOOD_IDX]}')")
            else:
                st.write(f"Base predicted class prob = {base_pgood:.4f}")
            
            sug_df = pd.DataFrame(suggestions)
            if not sug_df.empty:
                sug_df = sug_df.map(lambda x: round(x, 6) if isinstance(x, (int, float)) else x)
                st.write("Suggestions (sorted by improvement):")
                st.dataframe(sug_df[["feat","cur","best_val","delta","improvement","gain_per_unit"]])

            top = suggestions[0] if suggestions else None
            if top and top["improvement"] > 0:
                st.success(f"Top suggestion: เปลี่ยน {top['feat']} จาก {top['cur']} → {round(top['best_val'],4)} (Δ {round(top['delta'],4)}) -> P↑ {round(top['improvement'],4)}")
            else:
                st.warning("ไม่พบการปรับที่ทำให้ P(Good) เพิ่มขึ้นภายในช่วงที่กำหนด")
        except Exception as e:
            st.error(f"Error while generating suggestions: {e}")

st.write("---")
st.subheader("3) SHAP explanation (local)")
st.markdown("แสดงปัจจัยที่มีผลต่อการทำนาย P(Good) สำหรับคอยล์ปัจจุบัน")

if st.button("Compute SHAP for current profile"):
    with st.spinner("Calculating SHAP values... This may take a moment."):
        current_features = {**base_features, **cat_defaults}
        # ส่ง background data เข้าไปด้วย
        # fig, err = compute_and_plot_shap(current_features, X_train_sample)
        fig, err = compute_and_plot_shap(current_features)

        if err:
            st.error(err)
        else:
            good_class_name = "Good/No Defect"
            if GOOD_IDX is not None:
                good_class_name = label_encoder.classes_[GOOD_IDX]
            st.write(f"**Waterfall plot for class: '{good_class_name}'**")
            st.write("กราฟแสดงว่าแต่ละ feature 'ผลัก' ค่าความน่าจะเป็นออกจากค่าเฉลี่ย (base value) ไปในทิศทางบวก (เพิ่มโอกาส Good) หรือลบ (ลดโอกาส Good) มากน้อยเพียงใด")
            st.pyplot(fig)

st.write("---")
st.subheader("4) Human decision & Logging")
st.markdown("ถ้าคุณจะนำคำแนะนำไปใช้จริง ให้กรอกการปรับจริง (Actual adjustments) และผลลัพธ์ แล้วบันทึก เพื่อปิดลูปการเรียนรู้")

selected_feature = st.selectbox("เลือก feature ที่จะปรับ (หรือเลือก 'Manual')", ["Manual"] + list(actionable.keys()))
actual_changes = {}
if selected_feature != "Manual":
    suggested_val = st.number_input(f"ค่าสำหรับ {selected_feature} (Suggested)", value=float(base_features[selected_feature]))
    actual_changes[selected_feature] = st.number_input(f"ค่าสำหรับ {selected_feature} (Actual applied)", value=float(suggested_val))
else:
    for f in actionable.keys():
        actual_changes[f] = st.number_input(f"Actual applied for {f} (leave same if no change)", value=float(base_features[f]), key=f"actual_{f}")

operator = st.text_input("Operator / Engineer name", value="")
notes = st.text_area("Notes / observation")
outcome = st.selectbox("Outcome (observed)", ["Good", "Defect", "Other/Unknown"])

if st.button("Save decision and outcome to log"):
    # รวม features ทั้งหมดก่อนบันทึก
    all_base_features = {**base_features, **cat_defaults}
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "operator": operator,
        "base_features": json.dumps(all_base_features),
        "applied_features": json.dumps({k: float(actual_changes[k]) for k in actual_changes}),
        "selected_suggestion": selected_feature,
        "notes": notes,
        "outcome": outcome
    }
    append_log(entry)
    st.success("บันทึกสำเร็จ ➜ policy_log.csv")

if os.path.exists(LOGFILE):
    with open(LOGFILE,"rb") as f:
        st.download_button("ดาวน์โหลด log (policy_log.csv)", data=f, file_name=LOGFILE)