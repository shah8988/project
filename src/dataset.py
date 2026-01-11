
import numpy as np
import librosa
import os
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_features(sample):
    # Text embedding
    text_emb = text_model.encode(sample["caption"])

    # Audio path
    audio_path = sample.get("audio_path", None)

    if audio_path is None or not os.path.exists(audio_path):
        mfcc = np.zeros(20)
    else:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = np.mean(mfcc, axis=1)

    return np.concatenate([mfcc, text_emb])
