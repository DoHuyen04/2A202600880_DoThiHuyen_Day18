"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# Dùng paraphrase-multilingual-MiniLM-L12-v2 (~470MB, 384-dim) thay vì BAAI/bge-m3
# (~2.27GB, 1024-dim): cùng là embedding đa ngôn ngữ (hỗ trợ tiếng Việt) nhưng nhẹ
# hơn ~5x. Lý do: máy ít RAM không thể giữ đồng thời bge-m3 + cross-encoder reranker
# trong bộ nhớ khi chạy pipeline (gây segfault). Trên máy đủ RAM có thể đổi lại bge-m3
# (nhớ set EMBEDDING_DIM = 1024).
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
