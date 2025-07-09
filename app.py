import streamlit as st
import tensorflow as tf
import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("model/tokenizer_transformer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load SavedModel via TFSMLayer
@st.cache_resource
def load_model():
    return keras.layers.TFSMLayer("transformer_model", call_endpoint="serve")

# Parameter sesuai training
maxlen = 200

# Load model dan tokenizer
model = load_model()
tokenizer = load_tokenizer()

# Streamlit UI
st.title("🔍 Klasifikasi Ulasan Real vs Fake")
st.write("Masukkan teks ulasan produk. Model akan memprediksi apakah ulasan tersebut **Real** atau **Fake**.")

input_text = st.text_area("📝 Masukkan teks ulasan:", height=150)

if st.button("🔮 Prediksi"):
    if not input_text.strip():
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=maxlen).astype(np.float32)

        # Predict menggunakan TFSMLayer
        pred = model(padded)[0][0].numpy()
        label = "Fake" if pred >= 0.5 else "Real"

        st.markdown(f"### 🎯 Prediksi: **{label}** (Confidence: `{pred:.2f}`)")
        progress_value = float(pred if label == "Fake" else 1 - pred)
        st.progress(progress_value)