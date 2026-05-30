# Use the LATEST vLLM base image (Compatible with DeepSeek-R1 and PyTorch 2.4+)
FROM vllm/vllm-openai:latest

WORKDIR /

# Install RunPod and Sentence Transformers
RUN pip install runpod sentence-transformers huggingface_hub

# Copy the handler logic into the container
COPY handler.py /handler.py

# Pre-download text embedding models during the build phase
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Pre-download the AWQ Quantized DeepSeek-R1 Model
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='hugging-quants/DeepSeek-R1-Distill-Qwen-14B-AWQ-META')"

CMD [ "python3", "-u", "/handler.py" ]





# Force update