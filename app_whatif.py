# app_whatif.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from datetime import datetime

# plotting (plotly to avoid matplotlib dependency)
import plotly.express as px

# try import shap (optional) - if not available we'll skip SHAP but keep app usable
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="What-if: Surface Defect", layout="wide")

# --------------------------
# Load model & preprocessors
# --------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = tf.keras.models.load_model("surface_ann_model.keras")
    except Exception as e:
        st.error(f"Cannot load Keras model: {e}")
        raise

    try:
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("onehot_encoder.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
    except Exception as e:
        st.error(f"Cannot load preprocessing artifacts: {e}")
        raise

    return model, scaler, encoder, label_encoder

model, scaler, encoder, label_encoder = load_artifacts()

# --------------------------
# Feature lists (adjust if needed)
# --------------------------
num_cols = ['HNSPDI', 'WNSPDI', 'RMEXTG', 'SLFUTI',
            'LSP_Body', 'Entry_Body', 'XVPTF8', 'FT_HEAD',
            'CT_HEAD', 'FTGM', 'HDFBTH']
cat_cols = ['QUASTR', 'OPCCO', 'LCBXON', 'Product', 'ENDUSE', 'PASSNR']

# actionable numeric features to propose suggestions for
ACTIONABLE = ["CT_HEAD", "FT_HEAD", "XVPTF8", "FTGM", "HDFBTH", "LSP_Body", "Entry_Body"]

LOG_FILE = "whatif_log.csv"

# --------------------------
# Helpers
# --------------------------
def find_no_defect_index():
    """Try to find index of 'No Defect' (case-insensitive) in label_encoder.classes_.
       If not found, return default 3 only if valid, else return 0."""
    classes = list(label_encoder.classes_)
    for i, c in enumerate(classes):
        if isinstance(c, str) and "no" in c.lower() and ("defect" in c.lower() or "def" in c.lower() or "good" in c.lower()):
            return i
        if isinstance(c, str) and c.lower() in ("no defect", "nodefect", "no_defect", "good", "ok", "pass"):
            return i
    # fallback: if user previously said index 3, try that if exists
    if len(classes) > 3:
        return 3
    # last resort: class 0
    return 0

TARGET_CLASS_IDX = find_no_defect_index()

def preprocess_input(df_input):
    """
    Return X_all (scaled numeric + encoded categorical) ready for model.predict.
    Raises useful error if columns missing.
    """
    missing_num = [c for c in num_cols if c not in df_input.columns]
    missing_cat = [c for c in cat_cols if c not in df_input.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Missing columns. Numeric missing: {missing_num}, Cat missing: {missing_cat}")
    X_num = scaler.transform(df_input[num_cols])
    X_cat = encoder.transform(df_input[cat_cols]).toarray()
    X_all = np.hstack((X_num, X_cat))
    return X_all

def safe_predict(X_all):
    """Return preds (n_samples, n_classes) as numpy array"""
    preds = model.predict(X_all)
    preds = np.array(preds)
    # ensure 2D
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds

def compute_shap_values(X_all, preds, target_idx=TARGET_CLASS_IDX):
    """
    Compute SHAP values and return DataFrame and plotly figure for top features.
    This function is robust to different shapes returned by shap.
    """
    feature_names = []
    try:
        # build feature names: numeric first then encoded categorical feature names
        try:
            cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
        except Exception:
            # fallback generic names for encoded columns
            # encoder.categories_ may exist
            cat_feature_names = []
            try:
                for i, cats in enumerate(encoder.categories_):
                    for v in cats:
                        cat_feature_names.append(f"{cat_cols[i]}={v}")
            except Exception:
                cat_feature_names = [f"cat{i}" for i in range(encoder.transform([ ["" for _ in cat_cols] ]).shape[1])]
        feature_names = num_cols + cat_feature_names
    except Exception:
        feature_names = [f"f{i}" for i in range(X_all.shape[1])]

    # If shap not installed, raise informative error to caller
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap package not installed in environment. Install `shap` to enable SHAP explanation.")

    # Build a background with at least 2 rows to avoid some explainer issues
    if X_all.shape[0] < 2:
        background = np.vstack([X_all, X_all + np.random.normal(0, 1e-3, X_all.shape)])
    else:
        # small background sample (use X_all itself or jittered copies)
        background = X_all if X_all.shape[0] <= 50 else X_all[np.random.choice(X_all.shape[0], 50, replace=False)]

    # Try DeepExplainer first, fallback to KernelExplainer
    shap_values = None
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_all)
    except Exception:
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
        shap_values = explainer.shap_values(X_all, nsamples=50)

    # Normalize shap_values into a list of arrays, each [n_samples, n_features]
    if isinstance(shap_values, np.ndarray):
        # single array -> may correspond to one-output model. Convert to list of one array
        shap_list = [shap_values]
    elif isinstance(shap_values, (list, tuple)):
        shap_list = list(shap_values)
    else:
        # unexpected type
        shap_list = list(shap_values)

    # If predicted classes > len(shap_list), fallback to use argmax class or first available
    n_classes_pred = preds.shape[1]
    chosen_idx = target_idx if (0 <= target_idx < len(shap_list)) else min(len(shap_list)-1, np.argmax(preds))
    # final shap vector for sample 0
    try:
        shap_arr = shap_list[chosen_idx]  # shape (n_samples, n_features)
        # ensure shape
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)
        sv = shap_arr[0]
    except Exception:
        # ultimate fallback: zeros
        sv = np.zeros(X_all.shape[1])

    # Build DataFrame
    df_shap = pd.DataFrame({
        "Feature": feature_names[:X_all.shape[1]],
        "SHAP Value": sv,
        "Value": X_all[0]
    })
    df_shap["absSHAP"] = df_shap["SHAP Value"].abs()
    df_shap = df_shap.sort_values("absSHAP", ascending=False).drop(columns="absSHAP")

    # plot top 10 using plotly
    top_n = df_shap.head(10).sort_values("SHAP Value")
    fig = px.bar(top_n, x="SHAP Value", y="Feature", orientation='h',
                 title=f"Top SHAP impacts (class idx {chosen_idx})")
    return df_shap, fig

def suggest_changes(df_input, target_idx=TARGET_CLASS_IDX, pct_candidates=( -0.10, -0.05, 0.05, 0.10 )):
    """
    For each actionable numeric feature, try percentage changes in pct_candidates,
    and report the delta in P(target_idx). Returns list of suggestions sorted by improvement.
    Works in original feature space (df_input).
    """
    base_X = preprocess_input(df_input)
    base_preds = safe_predict(base_X)
    base_prob = float(base_preds[0, target_idx]) if target_idx < base_preds.shape[1] else float(base_preds[0].max())
    suggestions = []
    for feat in ACTIONABLE:
        cur_val = float(df_input.iloc[0][feat])
        best = {"feature": feat, "cur": cur_val, "best_val": cur_val, "best_prob": base_prob, "delta_pct": 0.0, "delta_prob": 0.0}
        for pct in pct_candidates:
            df_tmp = df_input.copy()
            df_tmp.at[0, feat] = cur_val * (1.0 + pct)
            try:
                X_tmp = preprocess_input(df_tmp)
                preds_tmp = safe_predict(X_tmp)
                new_prob = float(preds_tmp[0, target_idx]) if target_idx < preds_tmp.shape[1] else float(preds_tmp[0].max())
            except Exception:
                new_prob = base_prob
            if new_prob > best["best_prob"]:
                best.update({"best_val": df_tmp.at[0, feat], "best_prob": new_prob, "delta_pct": pct, "delta_prob": new_prob - base_prob})
        suggestions.append(best)
    # sort by absolute delta_prob descending
    suggestions_sorted = sorted(suggestions, key=lambda x: x["delta_prob"], reverse=True)
    return base_prob, suggestions_sorted

def append_log(entry, logfile=LOG_FILE):
    """Append entry dict to CSV"""
    row = entry.copy()
    row["timestamp"] = datetime.now().isoformat()
    if os.path.exists(logfile):
        df_log = pd.read_csv(logfile)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])
    df_log.to_csv(logfile, index=False)


# --------------------------
# Streamlit UI
# --------------------------
st.title("🔎 What-if Analysis: Surface Defect Prediction (ANN)")

st.markdown("ปรับค่า input ข้างล่างเพื่อดู P(No Defect) และคำแนะนำการปรับค่า (simulation)")

col1, col2 = st.columns(2)

with col1:
    input_data = {
        "HNSPDI": st.number_input("THICKNESS", value=4.0),
        "WNSPDI": st.number_input("WIDTH", value=1219.0),
        "RMEXTG": st.number_input("BAR_THICK", value=38.0),
        "SLFUTI": st.number_input("TIME_IN_FUR", value=3.5),
        "LSP_Body": st.number_input("LSP_Body", value=1100.0),
        "Entry_Body": st.number_input("Entry_Body", value=1040.0),
        "XVPTF8": st.number_input("SPEED", value=8.0),
    }
with col2:
    input_data.update({
        "FT_HEAD": st.number_input("FT_HEAD", value=860.0),
        "CT_HEAD": st.number_input("CT_HEAD", value=540.0),
        "FTGM": st.number_input("FM_FORCE", value=9000.0),
        "HDFBTH": st.number_input("HDFBTH", value=18.0),
        "QUASTR": st.selectbox("QUASTR", options=["C032", "C032RBB", "CG145", "CS0810", "CN1410", "CR1512"]),
        "OPCCO": st.selectbox("OPCCO", options=["0", "10", "21", "31", "41", "51", "66"]),
        "LCBXON": st.selectbox("LCBXON", options=["USED CB", "BYPASS CB"]),
        "Product": st.selectbox("PRODUCT", options=["ColdRoll", "CutSheet", "Other", "PO/POx", "Stock"]),
        "ENDUSE": st.selectbox("ENDUSE", options=["PNX", "SDX", "FXX", "DGX", "ADO", "ADH", "K1I", "GXX", "RST"]),
        "PASSNR": st.selectbox("RM_PASS", options=["5", "7", "9"])
    })

df_input = pd.DataFrame([input_data])

# Optional: debug info toggle
if st.checkbox("Show debug diagnostics (pred shapes, classes)"):
    st.write("Label classes:", list(label_encoder.classes_))
    try:
        Xtmp = preprocess_input(df_input)
        st.write("X_all shape (after preprocess):", Xtmp.shape)
    except Exception as e:
        st.write("Preprocess error:", str(e))

if st.button("🔮 Predict & Explain"):
    try:
        # preprocess + predict
        X_all = preprocess_input(df_input)
        preds = safe_predict(X_all)

        # show P(No Defect)
        n_classes = preds.shape[1]
        if TARGET_CLASS_IDX >= n_classes:
            st.warning(f"Target class index {TARGET_CLASS_IDX} out of range for prediction (n_classes={n_classes}). Showing argmax prob instead.")
            target_idx = int(np.argmax(preds))
        else:
            target_idx = TARGET_CLASS_IDX

        prob_no_defect = float(preds[0, target_idx])
        st.metric(label=f"P(No Defect)  (class idx {target_idx})", value=f"{prob_no_defect:.2%}")

        # Suggestions
        base_prob, suggestions = suggest_changes(df_input, target_idx=target_idx)
        st.subheader("📌 Suggested single-feature adjustments (simulation)")
        sug_df = pd.DataFrame(suggestions)
        if not sug_df.empty:
            st.dataframe(sug_df[["feature", "cur", "best_val", "delta_pct", "delta_prob", "best_prob"]].head(10))
        else:
            st.info("No suggestion found that increases P(No Defect) in tested candidates.")

        # SHAP (if available)
        st.subheader("📊 Local SHAP explanation")
        if SHAP_AVAILABLE:
            try:
                shap_df, fig = compute_shap_values(X_all, preds, target_idx=target_idx)
                st.dataframe(shap_df.head(10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
        else:
            st.info("SHAP not installed in environment. To enable SHAP, pip install shap and redeploy.")

        # append to log
        entry = {
            "input": str(input_data),
            "target_class_idx": int(target_idx),
            "P_NoDefect": float(prob_no_defect)
        }
        append_log(entry)

        st.success("Done — suggestion + SHAP shown (if available). Log appended to whatif_log.csv")

    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")
