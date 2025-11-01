import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
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
        .feature-line {
            text-align: center;
            font-size: 0.95rem;
            color: #AEEEEE;
            margin-top: 10px;
            margin-bottom: 20px;
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
st.markdown("<h3>🌊 Smart Fish Species Classifier</h3>", unsafe_allow_html=True)
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

# 🔹 Skip first 2 labels if model has 11 outputs (since you deleted 2 folders)
if len(labels) == 11:
    labels = labels[2:]

# ==============================
# DESCRIPTIONS
# ==============================
fish_descriptions = {
    "fish sea_food black_sea_sprat": "Small fish from the Black Sea, common in Mediterranean dishes.",
    "fish sea_food gilt_head_bream": "A silvery fish known for its mild and delicate flavor.",
    "fish sea_food hourse_mackerel": "Fast-swimming fish rich in omega-3 and found in coastal waters.",
    "fish sea_food red_mullet": "A reddish fish with sweet, tender flesh.",
    "fish sea_food red_sea_bream": "Common in warm seas, valued for its firm texture.",
    "fish sea_food sea_bass": "Popular fish with mild flavor and soft texture.",
    "fish sea_food shrimp": "Technically a crustacean, known for its sweet and juicy taste.",
    "fish sea_food striped_red_mullet": "Striped fish, often featured in fine-dining cuisines.",
    "fish sea_food trout": "Freshwater fish known for its vibrant spots and unique taste."
}

# ==============================
# FEATURES LINE (SINGLE LINE)
# ==============================
st.markdown("""
<div class='feature-line'>
🐟 Detects 9 Fish Species &nbsp;&nbsp;|&nbsp;&nbsp; 📊 Top-3 Predictions &nbsp;&nbsp;|&nbsp;&nbsp; 💬 Descriptions &nbsp;&nbsp;|&nbsp;&nbsp; 🌊 Blue-Themed Interface
</div>
""", unsafe_allow_html=True)

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
        try:
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not display image. Error: {e}")
            st.stop()

    with colB:
        try:
            img_array = preprocess_image(uploaded_file)
            preds = predict_image(model, img_array)[0]

            # Handle 11-output model (use last 9)
            if len(preds) == 11:
                preds = preds[2:]

            sorted_idx = np.argsort(preds)[::-1]
            top_idx = sorted_idx[:3]
            top_labels = [labels[i] for i in top_idx]
            top_scores = [preds[i] * 100 for i in top_idx]

            predicted_label = top_labels[0]
            confidence = top_scores[0]

            if confidence < 55:
                st.error("🚫 This doesn’t seem to be a clear fish image.")
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
            st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("⬆️ Upload an image to begin classification")
