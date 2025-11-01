import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
tflite = tf.lite
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🐟 FishVision", layout="centered", page_icon="🐠")

# ==============================
# CUSTOM STYLING (Blue Theme)
# ==============================
st.markdown("""
    <style>
        .main {
            background: linear-gradient(180deg, #001F3F 0%, #003366 40%, #0A1829 100%);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #E9F8FF;
            font-size: 2.3rem;
            margin-bottom: 0;
        }
        h3 {
            text-align: center;
            color: #B0E0E6;
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        .feature-box {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
            margin-bottom: 25px;
        }
        .feature {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 10px 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 0 8px rgba(0,191,255,0.3);
            font-size: 0.9rem;
        }
        .prediction-box {
            background: #04293A;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            color: #C7F9CC;
            font-size: 1.1rem;
            margin-top: 10px;
        }
        .desc {
            font-size: 0.95rem;
            color: #AEEEEE;
            text-align: center;
            margin-top: 6px;
            font-style: italic;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# HEADER SECTION
# ==============================
st.markdown("<h1>🐠 FishVision</h1>", unsafe_allow_html=True)
st.markdown("<h3>🌊 Intelligent Fish Species Classifier</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# LOAD MODEL + LABELS
# ==============================
MODEL_PATH = "fishvision_model.tflite"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

model = load_model()
labels = load_labels()

# ==============================
# SHORT DESCRIPTIONS
# ==============================
fish_descriptions = {
    "fish sea_food black_sea_sprat": "A small fish found in the Black Sea, often used in Mediterranean cuisine.",
    "fish sea_food gilt_head_bream": "A silvery fish known for its mild, delicate taste.",
    "fish sea_food hourse_mackerel": "Fast-swimming fish rich in omega-3 and found in coastal waters.",
    "fish sea_food red_mullet": "A reddish fish with a sweet, tender texture.",
    "fish sea_food red_sea_bream": "Common in warm seas, known for firm, white flesh.",
    "fish sea_food sea_bass": "Popular worldwide, loved for its soft flavor.",
    "fish sea_food shrimp": "Technically a crustacean, loved for its sweet, light taste.",
    "fish sea_food striped_red_mullet": "Striped fish often served in fine-dining dishes.",
    "fish sea_food trout": "Freshwater fish known for its vibrant spots and flavor.",
    "animal fish bass": "A strong freshwater predator popular in sport fishing.",
    "animal fish": "General aquatic species with varying traits."
}

# ==============================
# FEATURES SECTION
# ==============================
st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
st.markdown("<div class='feature'>🐟 Detects Multiple Fish Species</div>", unsafe_allow_html=True)
st.markdown("<div class='feature'>📊 Displays Top Predictions</div>", unsafe_allow_html=True)
st.markdown("<div class='feature'>💬 Provides Fish Descriptions</div>", unsafe_allow_html=True)
st.markdown("<div class='feature'>🌊 Compact Blue-Themed Interface</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# IMAGE UPLOAD
# ==============================
st.subheader("📸 Upload a Fish Image Below")
uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = Image.open(img).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ==============================
# PREDICTION SECTION
# ==============================
if uploaded_file:
    colA, colB = st.columns([1, 1])
    with colA:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    with colB:
        try:
            img_array = preprocess_image(uploaded_file)
            preds = predict_image(model, img_array)[0]
            sorted_idx = np.argsort(preds)[::-1]
            top_idx = sorted_idx[:3]

            top_labels = [labels[i] for i in top_idx]
            top_scores = [preds[i] * 100 for i in top_idx]

            predicted_label = top_labels[0]
            confidence = top_scores[0]

            if confidence < 65 or predicted_label.lower().startswith("animal"):
                st.error("🚫 This doesn’t seem to be a fish image.")
            else:
                st.markdown(
                    f"<div class='prediction-box'>🎯 <b>{predicted_label}</b><br>Confidence: {confidence:.2f}%</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p class='desc'>{fish_descriptions.get(predicted_label, 'A fascinating aquatic species!')}</p>",
                    unsafe_allow_html=True
                )

                st.markdown("### 🧠 Predictions Summary")
                for label, score in zip(top_labels, top_scores):
                    st.write(f"🐠 **{label}** — {score:.2f}%")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
else:
    st.info("⬆️ Upload an image to begin classification")
