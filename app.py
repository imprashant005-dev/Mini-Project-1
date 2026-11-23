import base64
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chest_disease_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")

st.set_page_config(page_title="Chest X-ray AI", page_icon="🫁", layout="centered")

# ---------------- Style ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp {
    background:
        radial-gradient(1100px 520px at 15% -10%, #0f4a5c 0%, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, #3a1f6e 0%, transparent 55%),
        #070b16;
    color: #e8eefc;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 880px; padding-top: 3.2rem; padding-bottom: 3rem; }

.stApp p, .stApp label, .stApp small, .stApp li { color: #cfd9f2; }

/* hero */
.hero { text-align: center; margin-bottom: 2.2rem; }
.pill {
    display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 0.78rem;
    letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
    color: #5eead4; background: rgba(94,234,212,0.10); border: 1px solid rgba(94,234,212,0.25);
}
.hero h1 {
    font-size: 3.1rem; line-height: 1.1; font-weight: 800; margin: 18px 0 10px 0; padding: 0;
    background: linear-gradient(90deg, #5eead4, #a78bfa);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { font-size: 1.05rem; color: #9fb0d6; margin: 0; }

/* uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04); border: 1.5px dashed rgba(255,255,255,0.22);
    border-radius: 20px; padding: 2.2rem 1rem; transition: all .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #5eead4; background: rgba(94,234,212,0.06); }
[data-testid="stFileUploaderDropzone"] * { color: #b6c4e6; }
[data-testid="stFileUploaderDropzone"] button {
    background: #5eead4; border: none; border-radius: 12px; font-weight: 700; padding: 8px 18px;
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #04222b; }
[data-testid="stFileUploaderFile"] * { color: #cfd9f2; }

/* result */
.glass {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px; padding: 26px 28px; backdrop-filter: blur(8px);
}
.xray { width: 100%; border-radius: 22px; border: 1px solid rgba(255,255,255,0.12); display: block; }
.label-sm { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #8aa0cc; font-weight: 600; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-top: 12px; font-size: 0.9rem; }
.bar-name { width: 118px; color: #b6c4e6; }
.bar-track { flex: 1; height: 8px; border-radius: 99px; background: rgba(255,255,255,0.08); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }
.bar-pct { width: 54px; text-align: right; color: #e8eefc; font-weight: 600; font-variant-numeric: tabular-nums; }
.note { font-size: 0.92rem; color: #9fb0d6; margin-top: 18px; line-height: 1.5; }
.warn { font-size: 0.88rem; color: #ffb020; margin-top: 12px; }
.chip {
    display: inline-block; margin: 4px 6px 4px 0; padding: 6px 12px; border-radius: 999px; font-size: 0.82rem;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); color: #cfd9f2;
}
.foot { text-align: center; font-size: 0.78rem; color: #6f81a9; margin-top: 3rem; }

[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; background: rgba(255,255,255,0.03); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #cfd9f2; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Data ----------------
CLASS_INFO = {
    "COVID": (
        "COVID-19",
        "#ff5c6c",
        "Hazy patches in both lungs, usually near the outer edges.",
    ),
    "Lung_Opacity": (
        "Lung Opacity",
        "#ffb020",
        "Cloudy, whiter areas in the lungs that are not COVID-19.",
    ),
    "Normal": (
        "Normal",
        "#35d07f",
        "The lungs look clear, with no obvious cloudy patches.",
    ),
    "Viral Pneumonia": (
        "Viral Pneumonia",
        "#a78bfa",
        "Cloudy areas spread through the lungs from a viral infection.",
    ),
}
LABELS = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CM = [[175, 9, 18, 0], [11, 161, 29, 1], [10, 16, 174, 2], [1, 0, 7, 193]]
F1 = {"COVID": 87.7, "Lung_Opacity": 83.0, "Normal": 80.9, "Viral Pneumonia": 97.2}
PRECISION = {
    "COVID": 88.8,
    "Lung_Opacity": 86.6,
    "Normal": 76.3,
    "Viral Pneumonia": 98.5,
}
MODEL_ACCURACY = 87.11
CLASS_ACCURACY = {
    "COVID": 86.63,
    "Lung_Opacity": 79.70,
    "Normal": 86.14,
    "Viral Pneumonia": 96.02,
}


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(image, dtype="float32"), axis=0)


def to_b64(image, max_side=720):
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- Hero ----------------
st.markdown(
    """
<div class="hero">
<span class="pill">Deep learning · DenseNet121</span>
<h1>Chest X-ray Analyzer</h1>
<p>Upload an X-ray. Get an instant prediction across four conditions.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    st.error(
        "Model files not found. Keep `chest_disease_model.keras` and `class_names.json` next to app.py."
    )
    st.stop()

model, class_names = load_model_and_classes()

uploaded = st.file_uploader(
    "Upload chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

# ---------------- Result ----------------
if uploaded is not None:
    image = Image.open(uploaded)
    with st.spinner("Analyzing..."):
        probs = model.predict(preprocess(image), verbose=0)[0]

    best = int(np.argmax(probs))
    title, color, meaning = CLASS_INFO.get(
        class_names[best], (class_names[best], "#5eead4", "")
    )
    conf = float(probs[best]) * 100

    st.write("")
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown(
            f'<img class="xray" src="data:image/jpeg;base64,{to_b64(image)}"/>',
            unsafe_allow_html=True,
        )

    with right:
        rows = ""
        for idx in np.argsort(-probs):
            n, c, _ = CLASS_INFO.get(
                class_names[idx], (class_names[idx], "#5eead4", "")
            )
            p_ = float(probs[idx]) * 100
            rows += (
                '<div class="bar-row">'
                f'<div class="bar-name">{n}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{max(p_, 1.5):.1f}%;background:{c};"></div></div>'
                f'<div class="bar-pct">{p_:.1f}%</div>'
                "</div>"
            )
        warn = (
            '<div class="warn">Low confidence. Treat this result with caution.</div>'
            if conf < 60
            else ""
        )
        st.markdown(
            f'<div class="glass">'
            f'<div class="label-sm">Prediction</div>'
            f'<div style="font-size:2.3rem;font-weight:800;color:{color};margin-top:6px;line-height:1.15;">{title}</div>'
            f'<div style="font-size:3.4rem;font-weight:800;color:#ffffff;line-height:1.1;margin-top:4px;">{conf:.1f}<span style="font-size:1.5rem;color:#8aa0cc;">%</span></div>'
            f'<div class="label-sm" style="margin-top:2px;">Confidence</div>'
            f'<div style="margin-top:18px;">{rows}</div>'
            f'<div class="note">{meaning}</div>'
            f"{warn}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------- About (collapsed) ----------------
st.write("")
with st.expander("About the model"):
    st.markdown(
        "Transfer learning with **DenseNet121**, trained on the COVID-19 Radiography Database. "
        "Test accuracy: **87.11%** on 807 unseen X-rays."
    )
    chips = "".join(
        f'<span class="chip"><b style="color:{CLASS_INFO[k][1]}">●</b> {CLASS_INFO[k][0]} · F1 {F1[k]}%</span>'
        for k in LABELS
    )
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    if os.path.exists(CM_IMAGE_PATH):
        st.image(CM_IMAGE_PATH, width=480)
    else:
        names = [CLASS_INFO[k][0] for k in LABELS]
        st.dataframe(pd.DataFrame(CM, index=names, columns=names))

st.markdown(
    '<div class="foot">:)</div>',
    unsafe_allow_html=True,
)
