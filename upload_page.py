# upload_page.py
import streamlit as st
from utils import get_pdf_text, get_text_chunks, get_vectorstore

def upload_page():
    st.header("Upload and Process Documents :robot_face:")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            
            st.session_state.vectorstore = vectorstore  # Save vectorstore for later use
            st.success("Documents processed successfully!")
