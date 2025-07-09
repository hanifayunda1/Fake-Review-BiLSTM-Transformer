import streamlit as st
import numpy as np
import pickle

from keras.models import load_model
from keras import layers
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences

# Define Custom Layer
class BiLSTMBlock(layers.Layer):
    def __init__(self, units, output_shape, dropout_rate=0.5, kernel_regularizer=0.01, bias_regularizer=0.01, **kwargs):
        super().__init__(**kwargs)
        self.bilstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
        self.dense1 = layers.Dense(output_shape,
                                   kernel_initializer='he_uniform',
                                   bias_initializer="zeros",
                                   kernel_regularizer=l2(kernel_regularizer),
                                   bias_regularizer=l2(bias_regularizer))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        self.dropout1 = layers.Dropout(dropout_rate)

        self.dense2 = layers.Dense(output_shape,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer="zeros",
                                   kernel_regularizer=l2(kernel_regularizer),
                                   bias_regularizer=l2(bias_regularizer))
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

# Load Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("model/BILSTMV4_model_2.h5", custom_objects={"BiLSTMBlock": BiLSTMBlock})
    with open("model/tokenizer_bilstm.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("Deteksi Spam Review dengan BiLSTM")
st.markdown("Masukkan review di bawah ini untuk memprediksi apakah termasuk **spam atau bukan**.")

user_input = st.text_area("✏️ Masukkan teks review:", height=150)

if st.button("🔍 Prediksi"):
    if not user_input.strip():
        st.warning("Harap masukkan teks terlebih dahulu.")
    else:
        # Tokenisasi dan padding
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=300)

        # Prediksi
        prediction = model.predict(padded)[0][0]
        label = "SPAM" if prediction > 0.5 else "BUKAN SPAM"

        st.markdown(f"### Hasil Prediksi: **{label}**")
        st.write(f"Probabilitas SPAM: {prediction:.4f}")