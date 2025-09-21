from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
model_name = "OuteAI/Lite-Oute-1-300M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model directly on CPU without device_map
model = AutoModelForCausalLM.from_pretrained(model_name)  # CPU by default

def run_agent(query, docs):
    """
    Args:
        query (str): User query
        docs (list of str): Relevant document chunks retrieved from MongoDB

    Returns:
        response (str): Generated answer with reasoning
    """
    context = "\n".join(docs)
    prompt = f"""
        Answer the query concisely using the context below.
        - Provide a clear, final answer.
        - Do NOT show reasoning steps.
        - Use bullet points only if needed.

        Context:
        {context}

        Query: {query}
        Answer:
    """


    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
