import streamlit as st
from api_config.api_config import configure_api_keys
from pdf_processor.pdf_processor import process_pdf
from vectorstore_manager.vectorstore_manager import create_retriever
from rag.rag import create_rag_chain, add_message
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

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
    """Main function for the Streamlit app."""
    st.title("📚 PDF RAG Chatbot")
    
    # Check if API key is configured
    if not configure_api_keys():
        return
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Document")
        
        # Upload file from streamlit file uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
         # Checking if file is valid, then generating a process buttong
        if uploaded_file is not None:
            process_button = st.button("Process PDF")
            
            # If the process button is true, `process_pdf` is called
            if process_button:
                success = process_pdf(uploaded_file)
                # If embeddings generated successfully, updating message and chat_history values in session_state
                if success:
                    st.session_state.messages = []
                    st.session_state.chat_history = ChatMessageHistory()
                    st.rerun()
            
            if st.session_state.pdf_processed:
                st.success(f"Currently using: {st.session_state.pdf_name}")
                # Clear messages, chat_history as new file is processed
                if st.button("Clear Chat History"):
                    st.session_state.messages = []
                    st.session_state.chat_history = ChatMessageHistory()
                    st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        # for each message, a context is created for each role
        with st.chat_message(message["role"]):
            # Message is printed for respective role
            st.write(message["content"])
    
    # Chat input, if the uploaded pdf is processed
    if st.session_state.pdf_processed:
        # Prompting user to enter a new query
        user_query = st.chat_input("Ask a question about your PDF...")
        
        if user_query:
            # Add user message 
            add_message("user", user_query)
            # st.write(st.session_state.messages)
            
            # Display user message
            with st.chat_message("user"):
                # Writing the query on screen
                st.write(user_query)
            
            # Create retriever to be used in rag_chain
            retriever = create_retriever()
            
            if retriever is not None:

                # complete rag chain
                rag_chain = create_rag_chain(retriever)
                
                # generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # generating response
                            response = rag_chain.invoke(user_query)
                            st.write(response)
                            
                            # Add AI message
                            add_message("assistant", response)
                            
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
            else:
                with st.chat_message("assistant"):
                    error_msg = "There was an issue with the document retrieval system. Please try processing the PDF again."
                    st.error(error_msg)
                    add_message("assistant", error_msg)
    else:
        st.info("Please upload a PDF document and click 'Process PDF' to start chatting.")
        

if __name__ == "__main__":
    main()
