# InsightEngine: Enterprise-Grade RAG System

**Live Demo:** <a href="https://insight-engine-lndychhed8zm5vkhn8nemp.streamlit.app/" target="_blank">https://insight-engine-lndychhed8zm5vkhn8nemp.streamlit.app/</a>

InsightEngine is a document retrieval system built to bridge the gap between static knowledge bases (PDFs) and conversational AI. 

Tackle the actual challenges of production AI: state management, conversational memory, and explainability. The goal was to create a system that doesn't just "chat" with a PDF, but allows for reliable, verifiable research.

## The Problem
Most basic RAG implementations have two critical failures:
1. **Amnesia:** They treat every query as a new interaction, failing when a user asks "Why?" or "Tell me more about that."
2. **Hallucination:** They give answers without proof, making them dangerous for enterprise use.

InsightEngine solves these by implementing a history-aware retrieval chain and forcing the model to cite its sources explicitly.

## System Architecture

The application follows a strict "Ingestion -> Retrieval -> Generation" pipeline designed for observability.

[Mermaid Diagram Placeholder - If you view this in a renderer that supports Mermaid]
graph LR
    A[PDF Document] -->|PyPDFLoader| B(Raw Text)
    B -->|Recursive Splitter| C(Semantic Chunks)
    C -->|OpenAI Embeddings| D[(ChromaDB)]
    
    User[User Query] -->|Contextualize| E{History Aware Chain}
    E -->|Rewrite Query| D
    D -->|Retrieve Top K| F[Context Documents]
    F -->|Stream| G[LLM Generation]
    G --> User

## Key Capabilities

### Context-Aware Memory
Instead of passing raw chat history to the LLM (which confuses the vector search), the system uses a secondary LLM call to rewrite the user's latest question. 
* *User:* "How much does it cost?"
* *System Internal:* "How much does [the payment processing module mentioned previously] cost?"
* *Result:* The vector database actually finds the right answer.

### Source Verification (Citations)
Trust is the biggest bottleneck for AI adoption. This engine decouples the retrieval step from generation. It presents the raw source segments (with page numbers) in an expandable UI component *before* the answer is fully generated, allowing the user to verify the bot's claims immediately.

### Streaming Latency
Waiting 5+ seconds for a complete answer feels broken. I implemented token-level streaming using Python generators, reducing the Time-To-First-Token (TTFT) to under 500ms, regardless of answer length.

## Getting Started

### Prerequisites
* Python 3.10+
* Docker (Optional, but recommended)
* OpenAI API Key

### Running Locally
1.  Clone the repo:
    git clone https://github.com/YOUR_USERNAME/insight-engine.git

2.  Install dependencies:
    pip install -r requirements.txt

3.  Set your API key in a `.env` file:
    OPENAI_API_KEY=sk-...

4.  Run the application:
    streamlit run app.py

### Running with Docker
docker-compose up --build

## Engineering Decisions

**Why Recursive Character Splitting?**
I chose `RecursiveCharacterTextSplitter` over simple token splitting because technical documents rely on structural context. Splitting blindly by token count often severs a header from its content. Recursive splitting respects paragraph and sentence boundaries, keeping semantic units intact.

**Why ChromaDB (Local)?**
For a portfolio/proof-of-work, a local persistent vector store reduces infrastructure overhead while still demonstrating how embedding storage works. The architecture is modular, so swapping `Chroma` for `Pinecone` or `Weaviate` in production would only require changing the connector in `rag_engine.py`.

## Future Improvements
* **Reranking:** Adding a Cross-Encoder step to score retrieved documents would significantly improve precision for niche queries.
* **Hybrid Search:** Combining vector search with keyword search (BM25) to better handle specific acronyms or ID numbers that semantic search sometimes misses.