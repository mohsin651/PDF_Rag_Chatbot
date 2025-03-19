# importing required libraries
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import shutil
import time

def remove_dir_with_retry(dir_path, retries=5, delay=1):
    # we are delaying because of different size of vector db created (this needs to be more optimized)
    for _ in range(retries):
        try:
            # removing the vector_db directory
            shutil.rmtree(dir_path)
            break  
        
        except PermissionError:
            time.sleep(delay)

def process_pdf(uploaded_file):
    """Process the uploaded PDF and create a vector store."""
    try:
         # Since PyPDFLoader expects a file path we are dumping the UploadedFile object from streamlit to tempfile to load via PyPDFLoader
         # Since we want to persist the file after end of with for splitting we set delete=False
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        # Load PDF
        with st.spinner("Loading PDF..."):
            # initializing the PyPDFLoader with file_path of temp file created
            pdf_loader = PyPDFLoader(file_path)
            # loading the temp file created with pdf_loader
            documents = pdf_loader.load()
            # adding the file name in session_state
            st.session_state.pdf_name = uploaded_file.name

        # Split documents
        with st.spinner("Processing document chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                # size of each chunk
                chunk_size=1000,
                # number of characters overlapping for each chunk
                chunk_overlap=200,
                # function for calculating number of chunks generated from the file
                length_function=len
            )
            
            # Perform PDF's spltting operation (this will return a list)
            doc_splits = text_splitter.split_documents(documents)
            # Add chunks to session_state for newer chats
            st.session_state.doc_splits = doc_splits
            # Print the number of chunks produced
            st.sidebar.success(f"Processed {len(doc_splits)} document chunks")

        # Create embeddings and vector database
        with st.spinner("Creating vector database..."):
            # Initialize embedding function
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create a unique persist directory for each session, where the VectorDB will be saved
            persist_dir = f"./chroma_db_{st.session_state.session_id}"
            
            
            # Remove any existing directory with the same name
            if os.path.exists(persist_dir):
                if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
                    # Ensure it's removed from session state
                    del st.session_state.vectorstore  

                time.sleep(1)  # Allow time for file handles to be released
                remove_dir_with_retry(persist_dir)  
            
            # Generate embedding via embedding and store in Chroma DB
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding_function,
                collection_name=f"pdf_collection_{st.session_state.session_id}",
                # Save in the persist directory
                persist_directory=persist_dir
            )
            
            # To maintain session awareness, updating vectorstore, persist_dir and changing pdf_processed key
            st.session_state.vectorstore = vectorstore
            st.session_state.persist_dir = persist_dir
            st.session_state.pdf_processed = True
            st.sidebar.success("Vector database created successfully!")

        # Clean up the temporary file as vector db is created
        os.unlink(file_path)

        # Return final state, signifying embeddings generated succesfully
        return True
        
    # Handle any potential error
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False
