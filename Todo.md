# 🗺️ InsightEngine: Technical Roadmap & Future Enhancements

This document outlines the architectural evolution of InsightEngine from a "Proof of Concept" (POC) to a production-ready Enterprise RAG system.

## 🚀 Phase 1: Retrieval Quality (The "Accuracy" Upgrade)
*Focus: Reducing hallucinations and improving precision for domain-specific queries.*

- [ ] **Hybrid Search Implementation**
    - **Current:** Dense Vector Search (Cosine Similarity).
    - **Upgrade:** Combine Vector Search with Keyword Search (BM25).
    - **Why:** Vector search struggles with exact matches (e.g., Part Numbers, SKUs, specific dates). Hybrid search covers this blind spot.
    
- [ ] **Reranking Layer (Cross-Encoders)**
    - **Plan:** Integrate `Cohere Rerank` or a local `BGE-Reranker`.
    - **Workflow:** Retrieve top 25 documents -> Rerank with high-fidelity model -> Pass top 5 to LLM.
    - **Why:** Drastically improves context relevance without increasing the context window cost.

- [ ] **Automated Evaluation Pipeline (RAGAS)**
    - **Plan:** Implement **RAGAS** (Retrieval Augmented Generation Assessment) or **DeepEval**.
    - **Metrics:** Automatically measure *Faithfulness*, *Answer Relevance*, and *Context Precision* on every commit.

## 🏗️ Phase 2: Infrastructure & Scalability
*Focus: Decoupling storage from compute and ensuring 99.9% uptime.*

- [ ] **Migration to Serverless Vector Storage**
    - **Current:** Local `ChromaDB` (Ephemeral).
    - **Upgrade:** `Pinecone Serverless` or `Weaviate Cloud`.
    - **Why:** Allows the knowledge base to persist across deployments and scale to millions of vectors without managing local disk state.

- [ ] **Session Persistence Layer**
    - **Current:** In-memory `st.session_state`.
    - **Upgrade:** Redis or PostgreSQL.
    - **Why:** Users can refresh the browser or switch devices without losing their chat history.

- [ ] **Observability & Tracing**
    - **Plan:** Integrate **LangSmith** or **Arize Phoenix**.
    - **Why:** To trace token usage, latency bottlenecks, and visualize the full chain of thought in production.

## 🎨 Phase 3: User Experience & Multimodality
*Focus: Expanding the ways users interact with data.*

- [ ] **Multi-Modal Ingestion**
    - **Plan:** Upgrade PDF loader to handle charts and images (using GPT-4 Vision or specialized OCR).
    - **Why:** Technical manuals often contain critical info in diagrams that text-only loaders miss.

- [ ] **"Chat with your Repository"**
    - **Plan:** Add a GitHub connector to ingest codebases (`.py`, `.js`, `.md`).
    - **Why:** Expands the use case from Document Analysis to Code Assistant.
