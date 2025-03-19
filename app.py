import streamlit as st
from api_config.api_config import configure_api_keys
from pdf_processor.pdf_processor import process_pdf
from vectorstore_manager.vectorstore_manager import create_retriever
from rag.rag import create_rag_chain, add_message
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def main():
    st.title("📖 PDF Information Retriever")
    
    if not configure_api_keys():
        return
    
    with st.sidebar:  
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.button("Process PDF"):
                if process_pdf(uploaded_file):
                    st.session_state.messages = []
                    st.session_state.chat_history = ChatMessageHistory()
                    st.rerun()

    if st.session_state.pdf_processed:
        user_query = st.chat_input("Ask a question about your PDF. (Please be specif in your query)")
        if user_query:
            add_message("user", user_query)
            retriever = create_retriever()
            if retriever:
                rag_chain = create_rag_chain(retriever)
                with st.chat_message("assistant"):
                    st.write(rag_chain.invoke(user_query))

if __name__ == "__main__":
    main()
