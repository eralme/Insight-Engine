import streamlit as st
import tempfile
import os
import shutil
from rag_engine import InsightEngine


# Page Config for a professional look
st.set_page_config(page_title="InsightEngine | AI Engineering Portfolio", layout="wide")

# 1. Initialize the Engine in Session State
# This ensures we don't re-instantiate the model/db on every rerun
if "engine" not in st.session_state:
    st.session_state.engine = InsightEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🚀 InsightEngine: Enterprise RAG")
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
        with st.spinner("Retrieving context..."):
            # Retrieve relevant chunks
            docs = st.session_state.engine.query_engine(prompt)

            # TODO: We will build the LLM Chain in the next step.
            # For now, let's show the retrieved context to verify the RAG logic.
            context_preview = "\n\n".join([f"**Source (Page {d.metadata.get('page', '?')}):** {d.page_content[:200]}..." for d in docs])
            response = f"I found {len(docs)} relevant segments. \n\n {context_preview}"

            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
