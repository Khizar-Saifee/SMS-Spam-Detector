
"""sms_glove_tf_pipeline.py
---------------------------------
End‑to‑end text‑classification demo that mixes:
• Gensim  – pre‑trained GloVe embeddings
• scikit‑learn – train/test split & metrics
• TensorFlow – small dense neural network

Dataset  : SMS Spam Collection (tab‑separated file: label \t text)
           Download from UCI or Kaggle and place as 'SMSSpamCollection'
Embedding: GloVe 6B 100‑d (auto‑download on first run)

Run:
    python sms_glove_tf_pipeline.py
"""

import os
import re
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_FILE = "C:/Users/DELL/Downloads/SMS Fraud Detector/SMSSpamCollection.csv"
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_TXT = "glove.6B.100d.txt"
W2V_FILE = "glove.6B.100d.w2v.txt"
EMBED_DIM = 100


def download_glove():
    if not os.path.exists(GLOVE_TXT):
        print("→ Downloading GloVe embeddings (128 MB)…")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
        print("→ Extracting", GLOVE_ZIP)
        with zipfile.ZipFile(GLOVE_ZIP, "r") as zf:
            zf.extract(GLOVE_TXT)
        os.remove(GLOVE_ZIP)


def load_embeddings():
    if not os.path.exists(W2V_FILE):
        download_glove()
        print("→ Converting GloVe text → word2vec format…")
        glove2word2vec(GLOVE_TXT, W2V_FILE)
    print("→ Loading embeddings into Gensim KeyedVectors…")
    return KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)


def clean(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.strip()


def vectorise(tokens, kv: KeyedVectors):
    vecs = [kv[w] for w in tokens if w in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM, dtype=np.float32)


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Download from "
            "https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection "
            "and place it in the same directory.")
    print("→ Loading dataset…")
    df = pd.read_csv(DATA_FILE)
    df['tokens'] = df['text'].map(lambda t: clean(t).split())

    kv = load_embeddings()

    print("→ Vectorising messages…")
    X = np.vstack([vectorise(toks, kv) for toks in df['tokens']])
    y = (df['label'] == 'spam').astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("→ Building & training model…")
    model = models.Sequential([
        layers.Input(shape=(EMBED_DIM,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=32,
              validation_split=0.1, verbose=2)
    # ✅ Save the trained weights
    model.save_weights("spam_model.weights.h5")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test accuracy: {acc * 100:.2f}%")
if __name__ == "__main__":
    main()



