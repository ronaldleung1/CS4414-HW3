#!/usr/bin/env python3

from encode import encode_query
from llm_generation import generate_response
from vector_db import load_documents, build_index, search

def search_documents(query, index, docs, top_k=3):
    query_embedding = encode_query(query)
    _, indices = search(index, query_embedding, top_k)
    
    retrieved_docs = []
    for idx in indices[0]:
        retrieved_docs.append(docs[idx]['text'])
    
    return retrieved_docs

def augment_prompt(query, retrieved_docs):
    prompt = query + " Top documents:"
    for doc in retrieved_docs:
        prompt += " " + doc
    return prompt

def main():
    docs = load_documents()
    index = build_index(docs)
    
    print("Type your question (or 'quit' to exit)\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not query:
            continue
        
        retrieved_docs = search_documents(query, index, docs, top_k=3)
        
        augmented_prompt = augment_prompt(query, retrieved_docs)
        response = generate_response(augmented_prompt)
        
        print("\nResponse:")
        print(response)
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()

