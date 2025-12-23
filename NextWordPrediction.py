import streamlit as st
import numpy as np
import pickle
import base64
import os
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from datetime import datetime

# ================================
# 📄 PAGE CONFIG
# ================================
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🧠",
    layout="centered"
)

# ================================
# 🎯 PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nwp_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ================================
# ⚙️ UTILITIES
# ================================
def get_base64_image_url(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")
    mime_type = uploaded_file.type or "image/png"
    return f"data:{mime_type};base64,{base64_encoded_data}"

def set_cinematic_bg(base64_urls, interval=6):
    if not base64_urls:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        return

    keyframes = []
    n = len(base64_urls)
    for i, url in enumerate(base64_urls):
        start = (i * 100) / n
        end = ((i + 1) * 100) / n
        keyframes.append(f"{start:.2f}% {{ background-image: url('{url}'); }}")
        keyframes.append(f"{end:.2f}% {{ background-image: url('{url}'); }}")

    st.markdown(f"""
    <style>
    .stApp {{
        background-size: cover;
        background-attachment: fixed;
        animation: bgAnim {n*interval}s infinite;
        color: white;
    }}

    @keyframes bgAnim {{
        {''.join(keyframes)}
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.65);
        z-index: 0;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }}

    * {{
        font-family: 'Poppins', sans-serif;
    }}
    </style>
    """, unsafe_allow_html=True)

# ================================
# 📂 SIDEBAR
# ================================
base64_images = []
with st.sidebar:
    st.title("🎨 App Settings")
    uploads = st.file_uploader(
        "Upload background images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploads:
        for img in uploads:
            base64_images.append(get_base64_image_url(img))
        st.success("Background applied")

    st.markdown("---")
    st.info("🧠 Next-Word Generator using LSTM")
    st.markdown(f"📅 {datetime.now().strftime('%b %d, %Y')}")
    st.markdown("👨‍💻 **Umar Imam**")

set_cinematic_bg(base64_images)

# ================================
# 🧠 LOAD MODEL
# ================================
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()
SEQ_LEN = model.input_shape[1] or 5

# ================================
# 🔥 SAMPLING
# ================================
def sample_with_temperature(preds, temperature=0.8):
    preds = np.log(preds + 1e-8) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))
    return np.random.choice(len(preds), p=preds)

def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0][-SEQ_LEN:]
    padded = pad_sequences([seq], maxlen=SEQ_LEN, padding="pre")
    preds = model.predict(padded, verbose=0)[0]
    idx = sample_with_temperature(preds)
    return tokenizer.index_word.get(idx, "<UNK>")

def generate_text(prompt, n):
    result = prompt.strip()
    for _ in range(n):
        word = predict_next_word(result)
        if word == "<UNK>":
            break
        result += " " + word
    return result

# ================================
# 🎓 HEADER
# ================================
st.markdown("""
<h1 style="text-align:center; color:#ff9900; text-shadow:2px 2px 6px #000;">
🧠 AI Text Generator
</h1>
<p style="text-align:center; font-size:18px;">
Generate movie-style text using deep learning
</p>
""", unsafe_allow_html=True)

# ================================
# 🧪 MAIN UI
# ================================
st.markdown("### ✍️ Starting Text")
prompt = st.text_input(
    "input",
    value="The movie was",
    label_visibility="collapsed"
)

st.markdown("### 🔢 Words to Generate")
num_words = st.slider(
    label="Number of words to generate",
    min_value=1,
    max_value=20,
    value=8,
    label_visibility="collapsed"
)

if st.button("🚀 Generate Text", use_container_width=True):
    with st.spinner("Generating..."):
        output = generate_text(prompt, num_words)

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.12);
        padding:30px;
        border-radius:18px;
        text-align:center;
        box-shadow:0 0 30px rgba(255,153,0,0.4);
    ">
        <h3 style="color:#ff9900;">Generated Text</h3>
        <p style="font-size:20px; line-height:1.6;">{output}</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# 📌 FOOTER
# ================================
st.markdown("""
    <div style="text-align:center; opacity:0.85;">
        Made with ❤️ &nbsp; | &nbsp; ✨ Developed by <b>Umar Imam</b>
    </div>
    """,
    unsafe_allow_html=True
)


