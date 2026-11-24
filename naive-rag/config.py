# Change these two lines if your Ollama version doesn't have llama4:scout.
# Fallback that exists on most installs: LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.1:8b"

DOCS_DIR = "docs"
STORE_FILE = "store.json"
CHUNK_WORDS = 300   # ~400 tokens
OVERLAP_WORDS = 50  # ~10-15% overlap
TOP_K = 5