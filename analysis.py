#!/usr/bin/env python3
"""
Part 3 Analysis Script - RAG Pipeline Performance Analysis
Measures latency breakdown for each component and explores optimizations
"""

import json
import time
import numpy as np
import faiss
from statistics import mean, stdev

# Import your existing modules
from encode import encode_query, load_model
from llm_generation import load_llm, generate_response
from vector_db import load_documents, build_index, search

def time_function(func, *args, **kwargs):
    """Helper to time a function call"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def run_component_analysis(queries, docs, index, num_runs=20):
    """
    Analyze latency breakdown for each component in the RAG pipeline
    """
    print("=" * 80)
    print("COMPONENT LATENCY ANALYSIS")
    print("=" * 80)
    
    # Pre-load models to avoid cold start in measurements
    print("\nPre-loading models...")
    load_model()
    load_llm()
    print("Models loaded.\n")
    
    encode_times = []
    search_times = []
    retrieval_times = []
    augment_times = []
    llm_times = []
    total_times = []
    
    test_queries = queries[:num_runs]
    
    print(f"Running {len(test_queries)} queries...\n")
    
    for i, q in enumerate(test_queries):
        query_text = q['text']
        
        # Total pipeline timing
        total_start = time.perf_counter()
        
        # 1. Encode query
        start = time.perf_counter()
        query_embedding = encode_query(query_text)
        encode_time = time.perf_counter() - start
        encode_times.append(encode_time * 1000)  # Convert to ms
        
        # 2. Vector search
        start = time.perf_counter()
        distances, indices = search(index, query_embedding, top_k=3)
        search_time = time.perf_counter() - start
        search_times.append(search_time * 1000)
        
        # 3. Document retrieval
        start = time.perf_counter()
        retrieved_docs = [docs[idx]['text'] for idx in indices[0]]
        retrieval_time = time.perf_counter() - start
        retrieval_times.append(retrieval_time * 1000)
        
        # 4. Prompt augmentation
        start = time.perf_counter()
        augmented_prompt = query_text + " Top documents:"
        for doc in retrieved_docs:
            augmented_prompt += " " + doc
        augment_time = time.perf_counter() - start
        augment_times.append(augment_time * 1000)
        
        # 5. LLM generation
        start = time.perf_counter()
        response = generate_response(augmented_prompt)
        llm_time = time.perf_counter() - start
        llm_times.append(llm_time * 1000)
        
        total_time = time.perf_counter() - total_start
        total_times.append(total_time * 1000)
        
        print(f"Query {i+1}/{len(test_queries)}: Total={total_time*1000:.1f}ms")
    
    print("\n" + "=" * 80)
    print("LATENCY BREAKDOWN (in milliseconds)")
    print("=" * 80)
    
    components = [
        ("Query Encoding", encode_times),
        ("Vector Search", search_times),
        ("Document Retrieval", retrieval_times),
        ("Prompt Augmentation", augment_times),
        ("LLM Generation", llm_times),
        ("TOTAL", total_times),
    ]
    
    results = {}
    for name, times in components:
        avg = mean(times)
        std = stdev(times) if len(times) > 1 else 0
        min_t = min(times)
        max_t = max(times)
        results[name] = {"mean": avg, "std": std, "min": min_t, "max": max_t, "raw": times}
        print(f"\n{name}:")
        print(f"  Mean: {avg:.2f} ms")
        print(f"  Std:  {std:.2f} ms")
        print(f"  Min:  {min_t:.2f} ms")
        print(f"  Max:  {max_t:.2f} ms")
    
    # Calculate percentage breakdown
    print("\n" + "=" * 80)
    print("PERCENTAGE BREAKDOWN (of total pipeline time)")
    print("=" * 80)
    total_avg = results["TOTAL"]["mean"]
    for name, times in components[:-1]:  # Exclude TOTAL
        pct = (results[name]["mean"] / total_avg) * 100
        print(f"{name}: {pct:.1f}%")
    
    return results


def run_topk_analysis(queries, docs, index, k_values=[1, 3, 5, 10]):
    """
    Analyze impact of different top-K values on search time and context size
    """
    print("\n" + "=" * 80)
    print("TOP-K ANALYSIS")
    print("=" * 80)
    
    load_model()
    load_llm()
    
    test_queries = queries[:10]
    results = {}
    
    for k in k_values:
        search_times = []
        llm_times = []
        context_lengths = []
        
        for q in test_queries:
            query_embedding = encode_query(q['text'])
            
            # Vector search timing
            start = time.perf_counter()
            distances, indices = search(index, query_embedding, top_k=k)
            search_times.append((time.perf_counter() - start) * 1000)
            
            # Build context
            retrieved_docs = [docs[idx]['text'] for idx in indices[0]]
            augmented_prompt = q['text'] + " Top documents:"
            for doc in retrieved_docs:
                augmented_prompt += " " + doc
            context_lengths.append(len(augmented_prompt))
            
            # LLM timing
            start = time.perf_counter()
            response = generate_response(augmented_prompt)
            llm_times.append((time.perf_counter() - start) * 1000)
        
        results[k] = {
            "search_ms": mean(search_times),
            "llm_ms": mean(llm_times),
            "context_chars": mean(context_lengths)
        }
        
        print(f"\nTop-K = {k}:")
        print(f"  Search time: {mean(search_times):.2f} ms")
        print(f"  LLM time: {mean(llm_times):.2f} ms")
        print(f"  Avg context length: {mean(context_lengths):.0f} chars")
    
    return results


def run_batch_analysis(queries, docs, batch_sizes=[1, 4, 8, 16, 32, 64, 128]):
    """
    Analyze batching at vector search step
    """
    print("\n" + "=" * 80)
    print("BATCH SIZE ANALYSIS (Vector Search)")
    print("=" * 80)
    
    # Load embeddings
    embeddings = np.array([doc['embedding'] for doc in docs]).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Prepare query embeddings
    load_model()
    query_embeddings = []
    print("Encoding queries for batch testing...")
    for q in queries:
        emb = encode_query(q['text'])
        query_embeddings.append(emb)
    query_embeddings = np.array(query_embeddings).astype('float32')
    
    results = {}
    
    for batch_size in batch_sizes:
        # Extend queries by cycling if batch_size > num queries
        if batch_size > len(query_embeddings):
            repeats = (batch_size // len(query_embeddings)) + 1
            extended_embeddings = np.tile(query_embeddings, (repeats, 1))
        else:
            extended_embeddings = query_embeddings
            
        times = []
        num_batches = max(1, len(query_embeddings) // batch_size)
        
        for i in range(min(num_batches, 10)):  # Run up to 10 batches
            start_idx = (i * batch_size) % len(extended_embeddings)
            batch = extended_embeddings[start_idx:start_idx + batch_size]
            # Handle wrap-around if needed
            if len(batch) < batch_size:
                batch = np.vstack([batch, extended_embeddings[:batch_size - len(batch)]])
            
            start = time.perf_counter()
            distances, indices = index.search(batch, 3)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        avg_time = mean(times)
        throughput = batch_size / (avg_time / 1000)  # queries per second
        latency_per_query = avg_time / batch_size
        
        results[batch_size] = {
            "batch_time_ms": avg_time,
            "per_query_ms": latency_per_query,
            "throughput_qps": throughput
        }
        
        print(f"\nBatch size = {batch_size}:")
        print(f"  Batch time: {avg_time:.3f} ms")
        print(f"  Per-query latency: {latency_per_query:.3f} ms")
        print(f"  Throughput: {throughput:.1f} queries/sec")
    
    return results


def run_index_comparison(docs, queries):
    """
    Compare FlatL2 vs IVFFlat search performance
    """
    print("\n" + "=" * 80)
    print("INDEX TYPE COMPARISON: FlatL2 vs IVFFlat")
    print("=" * 80)
    
    embeddings = np.array([doc['embedding'] for doc in docs]).astype('float32')
    dimension = embeddings.shape[1]
    n_docs = len(docs)
    
    # Build Flat index
    print("\nBuilding FlatL2 index...")
    start = time.perf_counter()
    flat_index = faiss.IndexFlatL2(dimension)
    flat_index.add(embeddings)
    flat_build_time = time.perf_counter() - start
    print(f"FlatL2 build time: {flat_build_time*1000:.2f} ms")
    
    # Build IVF index
    # nlist = number of clusters, typically sqrt(n) is a good starting point
    nlist = int(np.sqrt(n_docs))
    print(f"\nBuilding IVFFlat index (nlist={nlist})...")
    start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    ivf_index.train(embeddings)
    ivf_index.add(embeddings)
    ivf_build_time = time.perf_counter() - start
    print(f"IVFFlat build time: {ivf_build_time*1000:.2f} ms")
    
    # Prepare query embeddings
    load_model()
    test_queries = queries[:50]
    query_embeddings = np.array([encode_query(q['text']) for q in test_queries]).astype('float32')
    
    results = {"flat": {}, "ivf": {}}
    
    # Test Flat index
    print("\nTesting FlatL2...")
    flat_times = []
    for emb in query_embeddings:
        emb = emb.reshape(1, -1)
        start = time.perf_counter()
        D, I = flat_index.search(emb, 3)
        flat_times.append((time.perf_counter() - start) * 1000)
    
    results["flat"] = {
        "build_time_ms": flat_build_time * 1000,
        "search_mean_ms": mean(flat_times),
        "search_std_ms": stdev(flat_times)
    }
    print(f"FlatL2 search: {mean(flat_times):.3f} ± {stdev(flat_times):.3f} ms")
    
    # Test IVF index with different nprobe values
    for nprobe in [1, 4, 8, 16]:
        ivf_index.nprobe = nprobe
        ivf_times = []
        
        for emb in query_embeddings:
            emb = emb.reshape(1, -1)
            start = time.perf_counter()
            D, I = ivf_index.search(emb, 3)
            ivf_times.append((time.perf_counter() - start) * 1000)
        
        results["ivf"][nprobe] = {
            "search_mean_ms": mean(ivf_times),
            "search_std_ms": stdev(ivf_times)
        }
        print(f"IVFFlat (nprobe={nprobe}): {mean(ivf_times):.3f} ± {stdev(ivf_times):.3f} ms")
    
    results["ivf"]["build_time_ms"] = ivf_build_time * 1000
    
    # Accuracy comparison - check if IVF returns same results as Flat
    print("\nAccuracy comparison (IVF vs Flat ground truth):")
    for nprobe in [1, 4, 8, 16]:
        ivf_index.nprobe = nprobe
        matches = 0
        total = 0
        for emb in query_embeddings:
            emb = emb.reshape(1, -1)
            D_flat, I_flat = flat_index.search(emb, 3)
            D_ivf, I_ivf = ivf_index.search(emb, 3)
            for idx in I_ivf[0]:
                if idx in I_flat[0]:
                    matches += 1
                total += 1
        recall = matches / total * 100
        print(f"  nprobe={nprobe}: {recall:.1f}% recall@3")
    
    return results


def save_results(all_results, filename="analysis_results.json"):
    """Save results to JSON for plotting"""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    converted = convert(all_results)
    with open(filename, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    print("Loading documents and building index...")
    docs = load_documents()
    index = build_index(docs)
    
    with open('queries.json', 'r') as f:
        queries = json.load(f)
    
    print(f"Loaded {len(docs)} documents and {len(queries)} queries\n")
    
    all_results = {}
    
    # Run all analyses
    # all_results["component_latency"] = run_component_analysis(queries, docs, index, num_runs=20)
    # all_results["topk"] = run_topk_analysis(queries, docs, index, k_values=[1, 3, 5, 10])
    all_results["batching"] = run_batch_analysis(queries, docs)
    all_results["index_comparison"] = run_index_comparison(docs, queries)
    
    # Save results
    save_results(all_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

