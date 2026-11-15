
import streamlit as st
import requests
import os


# Page configuration
st.set_page_config(
    page_title="AI DevOps Assistant",
    page_icon="🤖",
    layout="centered"
)

# Backend API URL
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{BACKEND_URL}/chat"

# Application title
st.title("🤖 AI DevOps Assistant")
st.caption("Your personal AI assistant for DevOps, Cloud and Automation.")


# Initialize uploaded file content
if "file_content" not in st.session_state:
    st.session_state.file_content = ""

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

# File uploader
uploaded_file = st.file_uploader(
    "Upload PDF, LOG or TXT file",
    type=["pdf", "log", "txt"]
)

if uploaded_file is None:
    st.session_state.uploaded_file_id = None
    st.session_state.file_indexed = False
    st.session_state.document_id = None

else:
    file_id = (
        uploaded_file.name,
        uploaded_file.size,
        uploaded_file.getvalue()
    )

    if st.session_state.uploaded_file_id != file_id:
        st.session_state.file_indexed = False
        st.session_state.document_id = None
        st.session_state.uploaded_file_id = None
        try:
            with st.spinner("Reading file..."):
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue()
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()

                st.session_state.document_id = response.json()["document_id"]
                st.session_state.file_indexed = True
                st.session_state.uploaded_file_id = file_id

                st.success(f"File indexed successfully: {uploaded_file.name}")

        except requests.exceptions.RequestException as e:
            st.session_state.file_content = ""
            st.error(f"Upload failed: {e}")

    if st.session_state.get("file_indexed", False):
        st.success(f"File ready: {uploaded_file.name}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything about DevOps...")

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):

            try:
                # Use RAG when a file is uploaded
                
# Use RAG when a file is uploaded
                if st.session_state.get("file_indexed", False):
                    endpoint = f"{BACKEND_URL}/ask-document"

                    payload = {
                        "message": user_input,
                        "document_id": st.session_state.document_id
                    }

                else:
                    endpoint = API_URL

                    payload = {
                        "message": user_input
                    }

                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=180
                )


                response.raise_for_status()

                ai_reply = response.json()["reply"]

                st.markdown(ai_reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_reply
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to backend: {e}")

