import requests
import streamlit as st
from typing import List
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Tractian - Document Chatbot",
    page_icon="📚",
    layout="wide"
)

# Constants
API_URL = "http://backend:8000"  # FastAPI backend URL

def upload_files(uploaded_files: List) -> dict:
    """Upload files directly to the API."""
    if not uploaded_files:
        return None
    
    try:
        # Create files list for multipart/form-data
        files_to_upload = [
            ('files', (file.name, file.getvalue(), 'application/pdf'))
            for file in uploaded_files
        ]
                
        response = requests.post(
            f"{API_URL}/documents",
            files=files_to_upload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def ask_question(question: str) -> dict:
    """Send a question to the API."""
    try:
        response = requests.post(
            f"{API_URL}/question",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error asking question: {str(e)}")
        return None

# App title
st.title("📚 Document Chatbot")
st.markdown("Upload PDF documents and chat with them!")

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Upload files directly to API
                    result = upload_files(uploaded_files)
                    
                    if result:
                        st.success(f"Successfully processed {result['documents_indexed']} documents with {result['total_chunks']} chunks!")
                        st.session_state.documents_loaded = True
                    else:
                        st.error("Failed to process documents.")
                        st.session_state.documents_loaded = False
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.session_state.documents_loaded = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        if not st.session_state.documents_loaded:
            response_text = "Please upload and process some documents first!"
            st.warning(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            with st.spinner("Thinking..."):
                response = ask_question(prompt)
                
            if response:
                # Format the response with references
                answer = response["answer"]
                references = response["references"]
                
                response_text = f"{answer}\n\n"
                if references:
                    response_text += "\n\n**References:**\n"
                    for ref in references:
                        response_text += f"- {ref}\n"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                error_text = "Sorry, I couldn't process your question. Please try again."
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text}) 