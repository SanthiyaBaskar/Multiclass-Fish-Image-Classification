import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🐟 FishVision", layout="wide", page_icon="🐠")

# ==============================
# CUSTOM STYLING
# ==============================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #002b46 0%, #00446d 60%, #002b46 100%);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #E9F8FF;
            font-size: 2rem;
            margin-bottom: 0.3rem;
        }
        h3 {
            text-align: center;
            color: #AEEEEE;
            font-size: 1rem;
            margin-top: 0rem;
        }
        h2 {
            text-align: center;
            color: #FFD166;
            font-size: 1.1rem;
            margin-bottom: 0.6rem;
        }
        hr {
            border: 0.5px solid rgba(255,255,255,0.15);
            margin: 10px 0;
        }
        .feature-line {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .feature {
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 6px 12px;
            font-size: 0.85rem;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 0 6px rgba(0,191,255,0.25);
        }
        .prediction-box {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px;
            margin-top: 10px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("<h1>🐟🐠 FishVision 🐡🐠</h1>", unsafe_allow_html=True)
st.markdown("<h3>🌊 Smart Fish Species Classifier</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# LOAD MODEL & LABELS
# ==============================
MODEL_PATH = "fishvision_model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

model = load_model()
labels = load_labels()

# ==============================
# FISH DESCRIPTIONS
# ==============================
fish_descriptions = {
    "fish sea_food black_sea_sprat": "A small fish found in the Black Sea, popular in Mediterranean cuisine.",
    "fish sea_food gilt_head_bream": "A silver-scaled fish known for its mild, delicate flavor.",
    "fish sea_food hourse_mackerel": "A fast-swimming fish found in coastal waters, rich in omega-3.",
    "fish sea_food red_mullet": "A reddish fish known for its sweet taste and tender meat.",
    "fish sea_food red_sea_bream": "Commonly found in warm seas, valued for its firm texture.",
    "fish sea_food sea_bass": "A popular fish species found worldwide, prized for its flavor.",
    "fish sea_food shrimp": "Technically a crustacean, loved for its sweet and delicate taste.",
    "fish sea_food striped_red_mullet": "Recognized for its stripes and served in fine dining cuisines.",
    "fish sea_food trout": "A freshwater fish often found in cold streams and lakes.",
    "animal fish bass": "A strong freshwater predator known for its bold flavor.",
    "animal fish": "Generic fish category for aquatic species."
}

# ==============================
# HIGHLIGHTS
# ==============================
st.markdown("<h2>✨ Highlights</h2>", unsafe_allow_html=True)
st.markdown("""
<div class="feature-line">
    <div class="feature">🐟 Detects 11 Fish Species</div>
    <div class="feature">📊 Top-3 Predictions in Table</div>
    <div class="feature">💬 Gives Fish Description</div>
    <div class="feature">🌊 Ocean-Inspired Interface</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# IMAGE UPLOAD
# ==============================
st.subheader("📸 Upload Your Fish Image Below")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ==============================
# PREDICTION SECTION
# ==============================
if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="📷 Uploaded Image", width=230)

    with col2:
        img_array = preprocess_image(uploaded_file)
        preds = model.predict(img_array)[0]
        sorted_idx = np.argsort(preds)[::-1]
        top3_idx = sorted_idx[:3]
        top3_labels = [labels[i] for i in top3_idx]
        top3_scores = [preds[i]*100 for i in top3_idx]
        confidence = top3_scores[0]
        predicted_label = top3_labels[0]

        if confidence < 65 or predicted_label.lower().startswith("animal"):
            st.error("🚫 This doesn’t seem to be a fish image.")
        else:
            with st.container():
                st.markdown(f"### 🎯 Prediction: **{predicted_label}** ({confidence:.2f}%) 🐠🐡")
                st.markdown(f"*{fish_descriptions.get(predicted_label, 'A fascinating aquatic species!')}*")
                df = pd.DataFrame({
                    "🏅 Rank": [1, 2, 3],
                    "🐟 Fish Name": top3_labels,
                    "Confidence (%)": [f"{val:.2f}" for val in top3_scores]
                })
                st.table(df)
else:
    st.info("⬆️ Upload an image to begin classification")
