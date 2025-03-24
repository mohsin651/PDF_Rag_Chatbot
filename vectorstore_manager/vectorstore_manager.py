from langchain_chroma import Chroma
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


def create_retriever():
    """Safely create a retriever from the vectorstore."""
    try:
        # If we need to reload the vectorstore from disk, however if caching is not enabled, most like this loop will be skipped
        if st.session_state.vectorstore is None and st.session_state.pdf_processed:
            try:
                # Recreating the vectorDB
                # Get the embedding function
                embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Load the vectorstore from the persist directory
                persist_dir = st.session_state.persist_dir
                collection_name = f"pdf_collection_{st.session_state.session_id}"
                
                # vectordb creation
                vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embedding_function,
                    collection_name=collection_name
                )
                
                st.session_state.vectorstore = vectorstore
            except Exception as e:
                st.error(f"Error reloading vector database: {str(e)}")
                return None
        
        # Create the retriever
        if st.session_state.vectorstore is not None:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}) 
            return retriever
        
        return None
    except Exception as e:
        st.error(f"Error creating retriever: {str(e)}")
        return None
