# Latency Breakdown — Production RAG Pipeline

- Số chunks index: **107**
- Số query đo: **20**

## Build (one-time)

| Bước | Thời gian |
|------|-----------|
| 1. Chunking (M1) | 0.67 s |
| 2. Enrichment (M5) | 0.01 s |
| 3. Indexing BM25+Dense (M2) | 33.51 s |
| 4. Load reranker (M3) | 0.00 s |

## Per-query (trung bình)

| Bước | Latency (ms) | % |
|------|-------------:|---:|
| Retrieval (M2 BM25+Dense+RRF) | 60.1 | 1.2% |
| Rerank (M3 cross-encoder) | 3294.6 | 66.3% |
| LLM answer (gpt-4o-mini) | 1615.0 | 32.5% |
| **Tổng / query** | **4969.7** | **100%** |

## Evaluation

- RAGAS (4 metrics × 20 câu): **30.8 s**
