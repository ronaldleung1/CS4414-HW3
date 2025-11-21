#!/usr/bin/env python3

from llama_cpp import Llama

llm = None

def load_llm():
    global llm
    if llm is None:
        llm = Llama(
            model_path="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
            n_ctx=2048,
            verbose=False
        )
    return llm

def generate_response(prompt):
    model = load_llm()
    output = model(prompt, max_tokens=256, temperature=0.7)
    return output['choices'][0]['text']

