#!/usr/bin/env python3

import json
import numpy as np
import faiss

def load_documents():
    with open('preprocessed_documents.json', 'r') as f:
        docs = json.load(f)
    return docs

def build_index(docs):
    embeddings = np.array([doc['embedding'] for doc in docs]).astype('float32')
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search(index, query_embedding, top_k):
    query_embedding = query_embedding.astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def main():
    print("Loading preprocessed documents")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    
    print("Building FAISS index")
    index = build_index(docs)
    print(f"Index built with {index.ntotal} vectors (dimension: {index.d})")
    
    # Test - use document 42 as query
    test_doc_id = 42
    print(f"\nTest search using document {test_doc_id}")
    print(f"Text: \"{docs[test_doc_id]['text']}\"")
    
    query_embedding = np.array(docs[test_doc_id]['embedding'])
    k = 5
    distances, indices = search(index, query_embedding, k)
    
    print(f"\nTop {k} results:")
    print(f"{'Rank':<6} {'Doc ID':<10} {'Distance':<15} {'Text':<50}")
    print("-" * 90)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"{rank:<6} {idx:<10} {dist:<15.6f} {docs[idx]['text'][:50]}...")

if __name__ == "__main__":
    main()
