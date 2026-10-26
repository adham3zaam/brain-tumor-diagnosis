import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .header-container h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .header-container p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }

    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .result-tumor {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .result-normal {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    .result-card .confidence {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .result-card p {
        margin: 0.3rem 0 0 0;
        opacity: 0.95;
    }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
        padding: 0.8rem 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .prob-label {
        flex: 0 0 150px;
        font-weight: 600;
        color: #333;
    }
    .prob-bar-container {
        flex: 1;
        background: #e9ecef;
        border-radius: 10px;
        height: 24px;
        overflow: hidden;
        margin: 0 1rem;
    }
    .prob-bar {
        height: 100%;
        border-radius: 10px;
    }
    .prob-value {
        flex: 0 0 70px;
        text-align: right;
        font-weight: 700;
        color: #333;
    }

    /* Info card */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        color: #667eea;
    }
    .info-card p {
        margin: 0.3rem 0;
        color: #555;
        font-size: 0.95rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        color: #856404;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Config ----------
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "glioma": {
        "name": "Glioma",
        "description": "A tumor arising from glial cells in the brain or spine.",
        "severity": "Malignant"
    },
    "meningioma": {
        "name": "Meningioma",
        "description": "A tumor arising from the meninges (membranes surrounding the brain).",
        "severity": "Usually benign"
    },
    "notumor": {
        "name": "No Tumor",
        "description": "No tumor detected in the MRI scan.",
        "severity": "Normal"
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "description": "A tumor developing in the pituitary gland.",
        "severity": "Usually benign"
    }
}

CLASS_COLORS = {
    "glioma": "#ff6b6b",
    "meningioma": "#f39c12",
    "notumor": "#56ab2f",
    "pituitary": "#9b59b6"
}

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/brain_tumor_model.keras",
        compile=False
    )

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------- Header ----------
st.markdown("""
<div class="header-container">
    <h1>🧠 Brain Tumor MRI Diagnosis</h1>
    <p>AI-powered analysis for brain tumor detection and classification</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📋 About")
    st.markdown("""
    This application uses a deep learning model based on **MobileNetV2** 
    trained on brain MRI scans to detect and classify brain tumors.
    """)

    st.markdown("---")
    st.markdown("## 🎯 Detectable Classes")
    for key, info in CLASS_INFO.items():
        st.markdown(f"**{info['name']}**")
        st.caption(info['description'])

    st.markdown("---")
    st.markdown("## 📊 Model Info")
    st.markdown("- **Architecture:** MobileNetV2")
    st.markdown("- **Method:** Transfer Learning")
    st.markdown("- **Accuracy:** ~96%")
    st.markdown("- **Input Size:** 224×224")

# ---------- Main layout ----------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Reset state when new file uploaded
        file_id = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.get('last_file') != file_id:
            st.session_state['last_file'] = file_id
            for key in ['predictions', 'predicted_class', 'confidence']:
                st.session_state.pop(key, None)

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

with col2:
    st.markdown("### 🔬 Diagnosis")

    if uploaded_file is not None:
        if st.button("🔍 Run Diagnosis"):
            with st.spinner("Analyzing MRI scan..."):
                processed = preprocess_image(image)
                predictions = model.predict(processed, verbose=0)[0]
                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.session_state['predictions'] = predictions
            st.session_state['predicted_class'] = predicted_class
            st.session_state['confidence'] = confidence

    if 'predictions' in st.session_state:
        predicted_class = st.session_state['predicted_class']
        confidence = st.session_state['confidence']

        if predicted_class.lower() == "notumor":
            st.markdown(f"""
            <div class="result-card result-normal">
                <h2>✅ No Tumor Detected</h2>
                <div class="confidence">{confidence:.2f}%</div>
                <p>Confidence Level</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-tumor">
                <h2>⚠️ Tumor Detected</h2>
                <p style="font-size: 1.3rem; margin: 0.5rem 0; font-weight: 600;">
                    {CLASS_INFO.get(predicted_class, {}).get('name', predicted_class)}
                </p>
                <div class="confidence">{confidence:.2f}%</div>
                <p>Confidence Level</p>
            </div>
            """, unsafe_allow_html=True)

        if confidence < 60:
            st.warning("⚠️ Low confidence score. Professional medical review is recommended.")
    else:
        st.info("👈 Upload an MRI scan and click **Run Diagnosis** to see results.")

# ---------- Detailed Analysis ----------
if 'predictions' in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")

    predictions = st.session_state['predictions']
    sorted_probs = sorted(
        zip(class_names, predictions),
        key=lambda x: x[1],
        reverse=True
    )

    for name, prob in sorted_probs:
        color = CLASS_COLORS.get(name, "#667eea")
        display_name = CLASS_INFO.get(name, {}).get('name', name)
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{display_name}</div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: {prob*100}%; background: {color};"></div>
            </div>
            <div class="prob-value">{prob*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    predicted_class = st.session_state['predicted_class']
    info = CLASS_INFO.get(predicted_class, {})
    st.markdown(f"""
    <div class="info-card">
        <h4>ℹ️ About {info.get('name', predicted_class)}</h4>
        <p>{info.get('description', '')}</p>
        <p><strong>Severity:</strong> {info.get('severity', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)

