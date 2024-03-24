# chat_page.py
import streamlit as st
from utils import handle_userinput

def chat_page():
    st.header("Chat with PDFs :robot_face:")
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question)
