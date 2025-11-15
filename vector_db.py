#!/usr/bin/env python3

import json
import numpy as np
import faiss

def main():
    print("Loading preprocessed documents")
    with open('preprocessed_documents.json', 'r') as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents")
    
    print("Building FAISS index")
    embeddings = np.array([doc['embedding'] for doc in docs]).astype('float32')
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors (dimension: {dimension})")
    
    # Test - use document 42 as query
    test_doc_id = 42
    print(f"\nTest search using document {test_doc_id} as query:")
    print(f"Text: \"{docs[test_doc_id]['text'][:100]}\"")
    
    query_embedding = embeddings[test_doc_id:test_doc_id+1]
    k = 5
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nTop {k} results:")
    print(f"{'Rank':<6} {'Doc ID':<10} {'Distance':<15}")
    print("-" * 40)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        match = " <- SELF MATCH" if idx == test_doc_id else ""
        print(f"{rank:<6} {idx:<10} {dist:<15.6f}{match}")
    
    # Verification
    print("\nVerification:")
    if indices[0][0] == test_doc_id and distances[0][0] < 0.01:
        print("✓ PASS: Top result is query document with distance ≈ 0")
    else:
        print("✗ FAIL: Expected self-match with distance ≈ 0")

if __name__ == "__main__":
    main()
