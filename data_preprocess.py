#!/usr/bin/env python3

import json
from llama_cpp import Llama
import numpy as np

def main():
    print("Loading BGE model")
    model = Llama(
        model_path="bge-base-en-v1.5-f32.gguf",
        embedding=True,
        n_ctx=512,
        verbose=False
    )
    
    print("Loading documents")
    with open('documents.json', 'r') as f:
        docs = json.load(f)
    
    print(f"Encoding {len(docs)} documents")
    for i, doc in enumerate(docs):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(docs)}")
        embedding = model.embed(doc['text'])
        doc['embedding'] = embedding
    
    print("Saving preprocessed documents")
    with open('preprocessed_documents.json', 'w') as f:
        json.dump(docs, f, indent=2)

if __name__ == "__main__":
    main()

