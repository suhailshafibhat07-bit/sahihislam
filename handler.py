import runpod
import torch
import requests
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from vllm import LLM, SamplingParams
from app.database import hadith_collection

print("🚀 Initializing Enterprise Multilingual Models on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Multilingual BGE-M3 Embedder
bi_encoder = SentenceTransformer('BAAI/bge-m3', device=device)

# 2. Load Multilingual BGE-M3 Reranker
cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)

# 3. Load DeepSeek-R1-14B (AWQ Quantized) using vLLM for extreme speed
model_id = "hugging-quants/DeepSeek-R1-Distill-Qwen-14B-AWQ-META"
print(f"Loading LLM: {model_id}...")
llm = LLM(
    model=model_id,
    quantization="awq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.65, # Restricts LLM to 65% of VRAM to leave room for Embedders
    max_model_len=4096,
    trust_remote_code=True
)

print("✅ All models successfully loaded into VRAM!")

def handler(job):
    job_input = job.get("input", {})
    task = job_input.get("task")

    if not task:
        return {"error": "Missing 'task' parameter in input payload."}

    # --- TASK 1: VECTOR EMBEDDINGS ---
    if task == "embed":
        text = job_input.get("text", "")
        embedding = bi_encoder.encode(text, normalize_embeddings=True)
        return {"embedding": [float(x) for x in embedding]}

    # --- TASK 2: CROSS-ENCODER RERANKING ---
    elif task == "rerank":
        pairs = job_input.get("pairs", [])
        scores = cross_encoder.predict(pairs)
        return {"scores": [float(score) for score in scores]}

    # --- TASK 3: LLM GENERATION ---
    elif task == "generate":
        messages = job_input.get("messages", [])
        max_tokens = job_input.get("max_tokens", 800)
        
        # CORRECTED: DeepSeek-R1 / Qwen ChatML Template
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Trigger the assistant to start generating
        formatted_prompt += "<|im_start|>assistant\n"

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_tokens
        )
        
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return {"generated_text": generated_text}

    # --- TASK 4: REMOTE SEEDING (NEW) ---
    elif task == "seed":
        json_url = job_input.get("json_url")
        if not json_url:
            return {"error": "Missing 'json_url' in seed payload"}
        
        # Fetch data
        data = requests.get(json_url).json()
        texts = [doc.get("text_english", "") for doc in data]
        
        # Batch Embed on GPU
        embeddings = bi_encoder.encode(texts, batch_size=32, normalize_embeddings=True)
        
        for i, doc in enumerate(data):
            doc["embedding"] = [float(x) for x in embeddings[i].tolist()]
            
        # Perform DB insertion (Using asyncio.run to interface with async DB driver)
        asyncio.run(hadith_collection.insert_many(data, ordered=False))
        return {"status": "success", "inserted": len(data)}

    else:
        return {"error": f"Unknown task: '{task}'"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})