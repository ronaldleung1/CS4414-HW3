#!/usr/bin/env python3

from llama_cpp import Llama

llm = None

def load_llm():
    global llm
    if llm is None:
        llm = Llama(
            model_path="qwen2.5-0.5b-instruct-q5_k_m.gguf",
            n_ctx=2048,
            verbose=False
        )
    return llm

def generate_response(prompt):
    model = load_llm()
    output = model(prompt, max_tokens=256, temperature=0.7)
    return output['choices'][0]['text']

