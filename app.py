# app.py
import streamlit as st
from dotenv import load_dotenv
from upload_page import upload_page
from chat_page import chat_page
from utils import get_conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    menu = ["Upload & Process Documents", "Chat"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Upload & Process Documents":
        upload_page()
    elif choice == "Chat":
        if st.session_state.vectorstore is None:
            st.warning("Please upload and process documents first.")
        else:
            if st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
            chat_page()

if __name__ == '__main__':
    main()
