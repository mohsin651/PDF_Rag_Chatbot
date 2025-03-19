# PDF RAG Chatbot

## Overview
The **PDF RAG Chatbot** is a Streamlit-based application that allows users to upload PDF documents and interactively retrieve information using a **Retrieval-Augmented Generation (RAG)** approach. The chatbot leverages **LangChain**, **Groq's LLM (Llama-3.3-70B-SpecDec)**, and **ChromaDB** to process, store, and query PDF documents efficiently.

## Features
- Upload and process PDF files to extract text and store them in a **vector database**
- Use **HuggingFace Embeddings (all-MiniLM-L6-v2)** to generate vector representations of document chunks
- Retrieve context-relevant answers using a **retrieval-based query mechanism** (based on top 5 relevant answers)
- Maintain conversational context with LangChain's **ChatMessageHistory**
- Interactive chat interface built using **Streamlit**
- Uses **Groq API** to access a powerful large language model

## Directory Structure
```
mohsin651-pdf_rag_chatbot/
├── app.py
├── requirements.txt
├── api_config/
│   ├── __init__.py
│   ├── api_config.py
├── pdf_processor/
│   ├── __init__.py
│   ├── pdf_processor.py
├── rag/
│   ├── __init__.py
│   ├── rag.py
├── vectorstore_manager/
│   ├── __init__.py
│   ├── vectorstore_manager.py
```

- **app.py**: Main entry point for the Streamlit application  

- **api_config/**: Handles API key configuration  
 
- **pdf_processor/**: Processes and splits PDF documents  
 
- **rag/**: Implements retrieval-augmented generation logic  
 
- **vectorstore_manager/**: Manages vector database storage and retrieval  
  
## Installation

Follow these steps to install and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/mohsin651/pdf_rag_chatbot.git
cd pdf_rag_chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```
### 3. Install Dependencies
Install required Python libraries from requirements.txt:
```
pip install -r requirements.txt
```

### 4. Set Up API Key
The chatbot requires a Groq API Key to function:

- Obtain an API key from Groq from [here](https://console.groq.com/keys)
- Run the app and enter the API key in the sidebar

### 5. Running the Application
To launch the Streamlit app, run:
```
streamlit run app.py
```
Then, open the provided localhost URL in your browser.

### 6. How It Works

**Upload a PDF**: Select a PDF file to upload

**Processing**: The document is split into chunks, vectorized, and stored in ChromaDB

**Query the PDF**: Enter a question, and the chatbot retrieves relevant context to generate an answer

**Conversational Memory**: Maintains chat history for follow-up questions

**New File**: Previous ChatMemory is removed upon new PDF Processing

### 7. Troubleshooting

**Invalid API Key**: Ensure your API key is correct and set as an environment variable

**PDF Not Processing**: Check that the PDF format is supported and re-upload

**Slow Response Time**: Large PDFs may take longer to process; consider reducing chunk size and ChromaDB creation


### 8. Future Enhancements

This prototype demonstrates the core functionality of a PDF RAG chatbot, but there are several ways to enhance its performance and capabilities:

#### Advanced LLM Integration
- Upgrade to newer model versions as they become available
- Implement model switching to allow users to choose between different LLMs
- Fine-tune models on domain-specific data for improved performance

#### Enhanced Retrieval Techniques
- Implement re-ranking of retrieved documents to improve relevance
- Add hybrid search combining sparse and dense retrievers
- Incorporate semantic search with improved embedding models

#### Prompt Engineering
- Develop few-shot examples for different types of queries
- Create domain-specific prompt templates
- Implement dynamic prompt generation based on query type

#### User Experience
- Add document management features (save, delete, organize PDFs)
- Implement user authentication and personalization
- Create visualization tools for document insights

#### Scalability
- Optimize for larger documents and datasets
- Implement caching mechanisms for faster responses
- Add support for distributed processing of large document collections

#### Additional Features
- Multi-document querying
- Support for additional file formats (DOCX, TXT, etc.)
- Export conversation history and citations
- Implement metadata filtering and advanced search

### Acknowledgement

The code debugging was done with the help of ChatGPT, and comments were refined/corrected (but not generated) by an LLM.










