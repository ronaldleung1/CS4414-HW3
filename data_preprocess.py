#!/usr/bin/env python3

import json
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    with open('documents.json', 'r') as f:
        docs = json.load(f)
    
    texts = [doc['text'] for doc in docs]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    for i, doc in enumerate(docs):
        doc['embedding'] = embeddings[i].tolist()
    
    with open('preprocessed_documents.json', 'w') as f:
        json.dump(docs, f, indent=2)

if __name__ == "__main__":
    main()

