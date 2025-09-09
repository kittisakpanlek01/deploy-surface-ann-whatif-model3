# app_whatif.py  (หรือรวมเข้า app.py ของคุณ)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import datetime
import os
import json

# --- โหลด model & preprocessors (ปรับ path ตามจริง) ---
model = tf.keras.models.load_model("surface_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# features (ปรับให้ตรงกับของคุณ)
num_cols = ['HNSPDI','WNSPDI','RMEXTG','SLFUTI','LSP_Body','Entry_Body','XVPTF8','FT_HEAD','CT_HEAD','FTGM','HDFBTH']
cat_cols = ['QUASTR','OPCCO','LCBXON','Product','ENDUSE','PASSNR']

# ตัวแปรที่อนุญาตให้ปรับได้ (actionable)
actionable = {
    "CT_HEAD": {"min": 400, "max": 900, "step": 1},
    "FT_HEAD": {"min": 300, "max": 1200, "step": 1},
    "XVPTF8": {"min": 1.0, "max": 30.0, "step": 0.1},
    "FTGM": {"min": 1000, "max": 20000, "step": 10},
    "HDFBTH": {"min": 0.1, "max": 100.0, "step": 0.1},
    "LSP_Body": {"min": 800, "max": 1400, "step": 1},
    "Entry_Body": {"min": 800, "max": 1400, "step": 1},
}

LOGFILE = "policy_log.csv"

# --- helper: prepare X from a dict of features ---
def build_X_from_dict(features_dict):
    """Return X_all ready for model.predict (numpy array) given a dict with numeric+categorical."""
    # ensure numeric df ordered
    df_num = pd.DataFrame([{c: features_dict.get(c, np.nan) for c in num_cols}])
    df_cat = pd.DataFrame([{c: features_dict.get(c, "") for c in cat_cols}])
    # transform
    X_scaled = scaler.transform(df_num)
    X_cat_encoded = encoder.transform(df_cat).toarray()
    X_all = np.hstack((X_scaled, X_cat_encoded))
    return X_all, df_num, df_cat

# --- helper: get predicted probabilities + predicted label ---
def predict_probs(features_dict):
    X_all, df_num, df_cat = build_X_from_dict(features_dict)
    probs = model.predict(X_all)  # shape (1, n_classes)
    probs = probs.flatten()
    classes = label_encoder.inverse_transform(np.arange(len(probs)))
    prob_series = pd.Series(probs, index=classes)
    pred_class = classes[np.argmax(probs)]
    return prob_series, pred_class

# --- helper: find index for "Good/No Defect" class if possible ---
import re
def find_good_label_index():
    classes = list(label_encoder.classes_)
    for i, cls in enumerate(classes):
        if re.search(r"no[\s_-]*defect|no[\s_-]*def|good|ok|pass|non-defect", str(cls), re.I):
            return i
    return None

GOOD_IDX = find_good_label_index()  # may be None

# --- suggestion engine (simple simulation) ---
def suggest_changes(base_features, target_improve=0.05, n_steps=11, max_delta_fraction=0.1):
    """
    For each actionable numeric feature:
      - generate candidate values around current value (±max_delta_fraction)
      - compute P_good (or P_predicted) for each candidate (one-feature-at-a-time)
      - pick minimal delta that increases P_good by >= target_improve; if none, pick best improvement
    Returns list of suggestions sorted by expected gain per unit change.
    """
    base_prob_series, base_pred = predict_probs(base_features)
    if GOOD_IDX is not None:
        base_pgood = base_prob_series.iloc[GOOD_IDX]
    else:
        # fallback: use probability of predicted class if "good" target unknown
        base_pgood = base_prob_series[base_pred]

    suggestions = []
    for feat, cfg in actionable.items():
        cur = float(base_features.get(feat, np.nan))
        if np.isnan(cur):
            continue
        # candidate window
        span = max(1.0, abs(cur) * max_delta_fraction)
        candidates = np.linspace(cur - span, cur + span, n_steps)
        # clip to bounds
        candidates = np.clip(candidates, cfg["min"], cfg["max"])
        best = {"feat": feat, "cur": cur, "best_val": cur, "best_pgood": base_pgood, "delta": 0.0}
        for val in candidates:
            trial = base_features.copy()
            trial[feat] = float(val)
            prob_series, _ = predict_probs(trial)
            if GOOD_IDX is not None:
                pgood = prob_series.iloc[GOOD_IDX]
            else:
                # fallback use predicted class prob
                pgood = prob_series.max()
            if pgood > best["best_pgood"]:
                best.update({"best_val": val, "best_pgood": float(pgood), "delta": float(val-cur)})
        # compute improvement per unit change (avoid div by zero)
        improvement = best["best_pgood"] - base_pgood
        unit = abs(best["delta"]) if abs(best["delta"]) > 1e-9 else 1e-9
        gain_per_unit = improvement / unit
        suggestions.append({**best, "improvement": float(improvement), "gain_per_unit": float(gain_per_unit)})

    # sort suggestions by improvement (or gain_per_unit)
    suggestions_sorted = sorted(suggestions, key=lambda x: (x["improvement"], x["gain_per_unit"]), reverse=True)
    return base_prob_series, base_pgood, suggestions_sorted

# --- SHAP explanation (best-effort) ---
def compute_shap_values(base_features):
    """Try compute SHAP values. Returns dataframe of shap values or error string."""
    try:
        import shap
    except Exception as e:
        return None, f"shap not available: {e}"

    # Build background using a single baseline (could be improved by using real background dataset)
    try:
        X_all, df_num, df_cat = build_X_from_dict(base_features)
        # For keras model, try DeepExplainer when possible
        explainer = None
        try:
            explainer = shap.DeepExplainer(model, X_all)  # might fail if model incompatible
            shap_values = explainer.shap_values(X_all)
        except Exception:
            # fallback to KernelExplainer (slower)
            background = X_all  # tiny background
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
            shap_values = explainer.shap_values(X_all, nsamples=50)
        # shap_values may be list (per-class) or array
        return shap_values, None
    except Exception as e:
        return None, str(e)

# --- Logging function ---
def append_log(entry: dict):
    df_entry = pd.DataFrame([entry])
    if not os.path.exists(LOGFILE):
        df_entry.to_csv(LOGFILE, index=False)
    else:
        df_entry.to_csv(LOGFILE, mode="a", header=False, index=False)

# ----------------- Streamlit UI -----------------
st.title("What-If: Human-in-the-loop Recommendations")

st.write("ปรับค่า setpoints แบบทดลอง (What-If) และรับคำแนะนำ Δ ที่เป็นไปได้ พร้อม SHAP explanation และบันทึกการตัดสินใจของผู้ปฏิบัติงาน")

st.subheader("1) กรอกค่าปัจจุบันของคอยล์ (Current features)")
# create default base dict (you can load from file/DB in real app)
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
# other numeric features set to defaults or zeros
for c in num_cols:
    if c not in base_features:
        base_features[c] = st.number_input(c, value=0.0, key=f"num_{c}")

# categorical inputs (use first known categories from encoder if possible)
st.markdown("**Categorical features (use for model input)**")
cat_defaults = {}
col1, col2 = st.columns(2)
with col1:
    for c in ["QUASTR", "OPCCO", "LCBXON"]:
        default = "C032RBB" if c=="QUASTR" else ("31" if c=="OPCCO" else "USED CB")
        cat_defaults[c] = st.selectbox(c, options=encoder.categories_[c], index=0, key=f"cat_{c}", help=f"Select {c} category")
with col2:
    for c in ["Product", "ENDUSE", "PASSNR"]:
        default = "PO/POx" if c=="Product" else ("ADO" if c=="ENDUSE" else "5")
        cat_defaults[c] = st.selectbox(c, options=encoder.categories_[c], index=0, key=f"cat_{c}", help=f"Select {c} category")

# If no categories available, use text input
for c in cat_cols:
    # Try to set a sensible default if encoder has categories
    opts = None
    try:
        # encoder.categories_ corresponds to columns of encoder but may be for all categorical features combined
        # We cannot directly map easily in general; keep a text input fallback
        opts = None
    except Exception:
        opts = None
    if opts:
        cat_defaults[c] = st.selectbox(c, options=opts)
    else:
        cat_defaults[c] = st.text_input(c, value="")

# Predict current probabilities
if st.button("Predict current P(classes)"):
    try:
        probs, pred = predict_probs({**base_features, **cat_defaults})
        st.write("Probabilities per class:")
        st.dataframe(probs.to_frame("probability"))
        st.write("Predicted class:", pred)
        if GOOD_IDX is not None:
            st.info(f"P(good) estimated (class '{label_encoder.classes_[GOOD_IDX]}') = {probs.iloc[GOOD_IDX]:.4f}")
        else:
            st.info(f"No explicit 'good' class found; showing predicted-class prob = {probs.max():.4f}")
    except Exception as e:
        st.error(f"Predict error: {e}")

st.write("---")
st.subheader("2) ข้อเสนอแนะ (Suggest changes)")
target_improve = st.slider("เป้าหมายเพิ่ม P(Good) ขั้นต่ำ (absolute)", 0.0, 0.5, 0.05, 0.01)
if st.button("Generate suggestions"):
    with st.spinner("กำลังคำนวณข้อเสนอแนะ..."):
        try:
            base_prob_series, base_pgood, suggestions = suggest_changes({**base_features, **cat_defaults}, target_improve=target_improve)
            st.write("Base probabilities:")
            st.dataframe(base_prob_series.to_frame("prob"))
            if GOOD_IDX is not None:
                st.write(f"Base P(good) = {base_pgood:.4f} (class '{label_encoder.classes_[GOOD_IDX]}')")
            else:
                st.write(f"Base predicted class prob = {base_pgood:.4f}")

            # Show suggestions table
            sug_df = pd.DataFrame(suggestions)
            # format numbers
            for col in ["cur","best_val","improvement","delta","gain_per_unit"]:
                if col in sug_df.columns:
                    sug_df[col] = sug_df[col].map(lambda x: round(x,6))
            st.write("Suggestions (sorted by improvement):")
            st.dataframe(sug_df[["feat","cur","best_val","delta","improvement","gain_per_unit"]])

            # show top suggestion
            top = suggestions[0] if suggestions else None
            if top and top["improvement"]>0:
                st.success(f"Top suggestion: เปลี่ยน {top['feat']} จาก {top['cur']} → {round(top['best_val'],4)} (Δ {round(top['delta'],4)}) -> P↑ {round(top['improvement'],4)}")
            else:
                st.warning("ไม่พบการปรับที่ทำให้ P(Good) เพิ่มขึ้นภายในช่วงที่กำหนด")
        except Exception as e:
            st.error(f"Error while generating suggestions: {e}")

st.write("---")
st.subheader("3) SHAP explanation (local)")
if st.button("Compute SHAP for current profile (best-effort)"):
    shap_vals, shap_err = compute_shap_values({**base_features, **cat_defaults})
    if shap_err:
        st.error(f"SHAP not available: {shap_err}")
    else:
        st.write("SHAP computed (showing raw output). Note: visualization requires shap and JS support.")
        st.write(shap_vals)

st.write("---")
st.subheader("4) Human decision & Logging")
st.markdown("ถ้าคุณจะนำคำแนะนำไปใช้จริง ให้กรอกการปรับจริง (Actual adjustments) และผลลัพธ์ แล้วบันทึก เพื่อปิดลูปการเรียนรู้")

# Select a suggestion or manual adjustments
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
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "operator": operator,
        "base_features": json.dumps({k: float(base_features[k]) for k in base_features}),
        "applied_features": json.dumps({k: float(actual_changes[k]) for k in actual_changes}),
        "selected_suggestion": selected_feature,
        "notes": notes,
        "outcome": outcome
    }
    append_log(entry)
    st.success("บันทึกสำเร็จ ➜ policy_log.csv")

# allow download log
if os.path.exists(LOGFILE):
    st.download_button("ดาวน์โหลด log (policy_log.csv)", data=open(LOGFILE,"rb"), file_name=LOGFILE)
