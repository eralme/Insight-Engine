import streamlit as st
import tempfile
import os
import shutil
from rag_engine import InsightEngine

# BRIDGE: If running on Streamlit Cloud, map secrets to env vars for LangChain
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page Config for a professional look
st.set_page_config(page_title="InsightEngine | AI Engineering Portfolio", layout="wide")

# 1. Initialize the Engine in Session State
# This ensures we don't re-instantiate the model/db on every rerun
if "engine" not in st.session_state:
    st.session_state.engine = InsightEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("InsightEngine")
st.markdown("---")

# 2. Sidebar for Ingestion
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Knowledge Base", type="pdf")
    
    if st.button("Ingest & Vectorize"):
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Streamlit's UploadedFile is a BytesIO object. 
                # PyPDFLoader requires a file path, so we use a temp file.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                success = st.session_state.engine.ingest_document(tmp_path)
                os.remove(tmp_path) # Clean up
                
                if success:
                    st.success("Knowledge base updated!")
                else:
                    st.error("Ingestion failed. Check logs.")
    st.markdown("---")
    if st.button("⚠️ Reset Knowledge Base"):
        if os.path.exists("./chroma_db"):
            # 1. Clear the Vector Store from Memory
            # (Chroma's client can hold locks, so we try to force a release)
            st.session_state.engine.vector_store = None
            del st.session_state.engine
            
            # 2. Delete the Directory
            try:
                shutil.rmtree("./chroma_db")
                st.success("Database cleared!")
                
                # 3. Refresh the App to Re-initialize
                st.session_state.clear()
                st.rerun() 
            except Exception as e:
                st.error(f"Error clearing DB: {e}. \nTry stopping the app in terminal.")
        else:
            st.info("Database is already empty.")

# 3. Main Chat Interface
st.subheader("Query the Knowledge Base")

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a technical question about the uploaded docs..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Retrieval & Generation
    with st.chat_message("assistant"):
        # 1. Prepare History
        lc_history = []
        for msg in st.session_state.chat_history:
            role = "human" if msg["role"] == "user" else "ai"
            lc_history.append((role, msg["content"]))

        # 2. Retrieve Context First (Explicit Step)
        with st.spinner("Analyzing documents..."):
            context_docs = st.session_state.engine.retrieve_context(prompt, lc_history)

        # 3. Stream Answer
        response_stream = st.session_state.engine.generate_answer(prompt, context_docs, lc_history)
        full_response = st.write_stream(response_stream)

        # 4. Display Citations (The Enterprise Feature)
        with st.expander("📚 View Source Documents"):
            for i, doc in enumerate(context_docs):
                source_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i+1}:** {source_name} (Page {page_num})")
                st.caption(doc.page_content[:300] + "...")  # Preview first 300 chars
                st.divider()

        # 5. Save to History
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
