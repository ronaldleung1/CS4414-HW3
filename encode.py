#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np

model = None

def load_model():
    # use cached model, if it already exists
    global model
    if model is None:
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    return model

def encode_query(query_text):
    m = load_model()
    embedding = m.encode(query_text, convert_to_numpy=True)
    return embedding

