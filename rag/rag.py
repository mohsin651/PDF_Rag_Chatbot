import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from typing import List
from langchain_core.documents import Document

def initialize_llm():
    """Initialize the LLM with Groq."""
    # returnign ChatGroq object with model's specifics
    return ChatGroq(
        model_name="llama-3.3-70b-specdec",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=2,
    )

def doc2str(docs: List[Document]) -> str:
    """Convert document chunks to a single string."""
    if not docs:
        return "No relevant information found in the document."
    # joining all the relevant docs' page_content in a formatter manner
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    """Create the RAG chain with the retriever."""
    # Create the prompt template
    template = """
    Answer the question strictly based on the provided context. 
    If the context does not contain enough information to provide a confident answer, respond with: 'I don't have enough information to answer this question based on the given context, 
    and ask the user to enter specific query related to the pdf' 
    However, if you can answer the question using general knowledge, provide a response afterward, clearly indicating that it is not derived from the provided context. 
    Be kind and respectful in your response."
    
    Previous conversation:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize LLM (ChatGroq)
    llm = initialize_llm()
    
    # Create the chain
    rag_chain = ( # The recieved question will be passed to all the values in the dictionary below
        {
            "context": lambda x: get_context(retriever, x),
            "question": RunnablePassthrough(),
            # since the past questioned will be passed to the chat_history as well, since we don't need, thus it is dumped
            "chat_history": lambda _: get_chat_history_str()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def get_context(retriever, query):
    """retrieve context and handle potential errors."""
    try:
        # retreiving relevant chunks
        docs = retriever.invoke(query)
        
        # returning the page_contents of retrieved chunks
        return doc2str(docs)

    # handling  any potential error
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving relevant information from the document."


def get_chat_history_str():
    """Convert chat history to a formatted string."""
    # Initializing if it chat_history is not listed in session_state
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = ChatMessageHistory()
        return ""
    
    # initialzing the defaulty value of chat history
    history_messages = st.session_state.chat_history.messages
    history_str = ""
    
    # adding respective messages to chat history
    for message in history_messages:
        if isinstance(message, HumanMessage):
            history_str += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            history_str += f"AI: {message.content}\n"
    
    # returning the previous human/ai messages in str format
    return history_str


def add_message(role, content):
    """Add a message to session state and chat history."""
    # This is for displaying past queries and responses
    st.session_state.messages.append({"role": role, "content": content})
    
    # For maintaining conversational context for follow up questions
    if role == "user":
        st.session_state.chat_history.add_user_message(content)
    else:
        st.session_state.chat_history.add_ai_message(content)