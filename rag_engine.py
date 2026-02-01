import os
import logging
from typing import List, Optional

# Streamlit
import streamlit as st

# Langchain Core & Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Stores and Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Schema & Interfaces
from langchain_core.documents import Document

# History Awarehttps://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportMissingImports.md
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightEngine:
    def __init__(self, collection_name: str = "insight_engine_core"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = "./chroma_db"
        self.collection_name = collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Chroma:
        """Initializes or loads the local ChromaDB instance."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def ingest_document(self, file_path: str) -> bool:
        """
        Production pipeline: Load -> Split -> Embed -> Store.
        Includes basic error handling and metadata tracking.
        """
        try:
            logger.info(f"Starting ingestion for: {file_path}")
            
            # 1. Document Loading
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            # 2. Text Splitting
            # Note: We keep metadata from PyPDF (page numbers, etc.)
            chunks = self.text_splitter.split_documents(raw_docs)
            
            # 3. Embedding & Storage
            self.vector_store.add_documents(chunks)
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            return False

    def query_engine(self, query: str, k: int = 5) -> List[Document]:
        """Exposes the retriever interface for the Streamlit UI."""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_streaming_response(self, query: str, chat_history: List[dict]):
        """
        Creates a chain that:
        1. Reformulates the question based on history (Contextualize).
        2. Retrieves documents.
        3. Streams the answer.
        """
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, streaming=True)

        # 1. Contextualize Question Prompt
        # This helps the LLM understand "it" refers to the previous topic
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context
        in the chat history, formulate a standalone question which can be understood without the chat history. DO NOT answer the question,
        just formulate it if needed and otherwise return it as is.
        """
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
                llm, self.vector_store.as_retriever(), contextualize_q_prompt
            )

        # 2. Answer Prompt
        qa_system_prompt = """You are an assistant for Question-Answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
    
        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 3. Execute & Stream
        # We process the chat_history from Streamlit format to LangChain format if needed
        response = rag_chain.stream({"input": query, "chat_history": chat_history})
        
        for chunk in response:
            if "answer" in chunk:
                yield chunk["answer"]

    def retrieve_context(self, query: str, chat_history: List[dict]) -> List[Document]:
        """
        Retrieves relevant documents using the history-aware logic.
        Detailed for debugging retrieval quality.
        """
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # Same contextualization logic as before
        contextualize_q_system_prompt = """Given a chat history and the latest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, self.vector_store.as_retriever(), contextualize_q_prompt
        )

        # Execute retrieval only
        return history_aware_retriever.invoke({"input": query, "chat_history": chat_history})

    def generate_answer(self, query: str, context: List[Document], chat_history: List[dict]):
        """
        Generates the answer using the PRE-RETRIEVED context.
        """
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, streaming=True)

        qa_system_prompt = """You are an assistant for Question-Answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.

        Context:
        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # We use a simple chain here because we already have the docs
        chain = create_stuff_documents_chain(llm, qa_prompt)

        # Stream the output
        for chunk in chain.stream({"context": context, "input": query, "chat_history": chat_history}):
            # 'create_stuff_documents_chain' output structure is simpler
            yield chunk
