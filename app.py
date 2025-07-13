import streamlit as st
import numpy as np
import pandas as pd
import re
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
EMBED_DIM = 100
W2V_FILE = "glove.6B.100d.w2v.txt"

@st.cache_resource
def load_embeddings():
    return KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.strip()

def vectorise(tokens, kv):
    vecs = [kv[word] for word in tokens if word in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM, dtype=np.float32)

@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Input(shape=(EMBED_DIM,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("spam_model.weights.h5")
    return model

st.title("📩 SMS Spam Classifier")
st.write("Enter a text message and this app will classify it as **Spam** or **Not Spam** using GloVe embeddings + a neural network.")

user_input = st.text_area("✉️ Enter your message:", height=150)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        kv = load_embeddings()
        model = load_model()
        tokens = clean(user_input).split()
        X = vectorise(tokens, kv).reshape(1, -1)
        prediction = model.predict(X)[0][0]
        if prediction >= 0.5:
            st.error(f"❌ SPAM — Confidence: {prediction:.2f}")
        else:
            st.success(f"✅ NOT SPAM — Confidence: {1 - prediction:.2f}")