import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.regularizers import l2

# Streamlit Option Menu
st.set_page_config(page_title="Fake Review Detector", layout="centered")
selected = option_menu(
    menu_title="Menu",
    options=["Home", "Transformer Model", "BiLSTM Model"],
    icons=["house", "cpu", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# BiLSTM Custom Layer
class BiLSTMBlock(layers.Layer):
    def __init__(self, units, output_shape, dropout_rate=0.5, kernel_regularizer=0.01, bias_regularizer=0.01, **kwargs):
        super().__init__(**kwargs)
        self.bilstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
        self.dense1 = layers.Dense(output_shape, kernel_initializer='he_uniform', bias_initializer="zeros",
                                   kernel_regularizer=l2(kernel_regularizer), bias_regularizer=l2(bias_regularizer))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer="zeros",
                                   kernel_regularizer=l2(kernel_regularizer), bias_regularizer=l2(bias_regularizer))
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation("sigmoid")

    def call(self, inputs, training=None):
        x = self.bilstm(inputs)
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        return self.act2(x)

# Home Page
if selected == "Home":
    st.markdown("## 🛍️ Fake Review Detector for E-Commerce")
    st.markdown("""
    Selamat datang di aplikasi **Fake Review Detector**!

    Aplikasi ini dirancang untuk **mendeteksi ulasan palsu (fake review)** pada platform e-commerce, 
    yang sering digunakan untuk menipu konsumen atau memanipulasi reputasi produk.

    ---  
    🔍 **Tersedia dua model klasifikasi**:
    - **Transformer Model**: menggunakan arsitektur Transformer modern
    - **BiLSTM Model**: menggunakan pendekatan Bi-directional LSTM klasik

    Anda dapat menguji ulasan secara langsung dan melihat prediksinya apakah review tersebut asli (**Real**) atau palsu (**Fake**).

    Silakan pilih tab di atas untuk mencoba model deteksi.
    """)

# Transformer Model Page
elif selected == "Transformer Model":
    st.markdown("## ⚡ Deteksi Fake Review with Transformer")

    @st.cache_resource
    def load_transformer_model():
        return keras.layers.TFSMLayer("model/transformer_model", call_endpoint="serve")

    @st.cache_resource
    def load_transformer_tokenizer():
        with open("model/tokenizer_transformer.pkl", "rb") as f:
            return pickle.load(f)

    transformer_model = load_transformer_model()
    transformer_tokenizer = load_transformer_tokenizer()

    maxlen = 200
    input_text = st.text_area("📝 Masukkan ulasan produk:", height=150)

    if st.button("Prediksi (Transformer)"):
        if not input_text.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            sequence = transformer_tokenizer.texts_to_sequences([input_text])
            padded = pad_sequences(sequence, maxlen=maxlen).astype(np.float32)

            pred = transformer_model(padded)[0][0].numpy()
            label = "Fake Review" if pred >= 0.5 else "Real Review"

            st.markdown(f"### 🎯 Prediksi: **{label}** (Confidence: `{pred:.2f}`)")
            progress_value = float(pred if label == "Fake Review" else 1 - pred)
            st.progress(progress_value)

# BiLSTM Model Page
elif selected == "BiLSTM Model":
    st.markdown("## 🧠 Deteksi Fake Review with BiLSTM")

    @st.cache_resource
    def load_bilstm_model_and_tokenizer():
        model = keras.models.load_model(
            "model/BILSTMV4_model_2.h5",
            custom_objects={"BiLSTMBlock": BiLSTMBlock}
        )
        with open("model/tokenizer_bilstm.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer

    bilstm_model, bilstm_tokenizer = load_bilstm_model_and_tokenizer()
    input_text = st.text_area("✏️ Masukkan ulasan produk:", height=150, key="bilstm_input")

    if st.button("Prediksi (BiLSTM)"):
        if not input_text.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            sequence = bilstm_tokenizer.texts_to_sequences([input_text])
            padded = pad_sequences(sequence, maxlen=300)
            prediction = bilstm_model.predict(padded)[0][0]
            label = "Fake Review" if prediction > 0.5 else "Real Review"

            st.markdown(f"### 🎯 Prediksi: **{label}** (Confidence: `{prediction:.2f}`)")
            st.progress(float(prediction if label == "Fake Review" else 1 - prediction))
