import os
import streamlit as st

def configure_api_keys():
    """Configure API keys and environment variables."""
    # Check if API key is in session state, otherwise use the input form
    if "groq_api_key" not in st.session_state or not st.session_state.groq_api_key:
        st.sidebar.title("API Configuration")
        api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
        if api_key:
            st.session_state.groq_api_key = api_key
            os.environ["GROQ_API_KEY"] = api_key
            return True
        else:
            st.sidebar.warning("Please enter a Groq API key to continue.")
            return False
    else:
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
        return True