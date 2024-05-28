# upload_page.py
import streamlit as st
from utils import get_pdf_text, get_text_chunks, get_vectorstore
from langchain.embeddings import HuggingFaceEmbeddings
import mysql.connector
from langchain.vectorstores import FAISS
from config import sql_user, sql_pass


def load_index_from_database(identifier):
    conn = mysql.connector.connect(
        host='localhost',
        user=sql_user,
        password=sql_pass,
        database='stars'
    )
    cursor = conn.cursor()
    print('Connecting..')

    # Retrieve the serialized index from the database
    cursor.execute("SELECT vector_store FROM vector_stores WHERE id = %s", (identifier,))
    print('Retrieving..')
    row = cursor.fetchone()

    conn.close()
    if row:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        serialized_index = row[0]
        vector_store = FAISS.deserialize_from_bytes(
            embeddings=embeddings, serialized=serialized_index
        ) 

        # Deserialize the index
        # vector_store = faiss.deserialize_index(serialized_index)

        return vector_store
    else:
        print("No vector store found with the given identifier.")
        return None


def upload_page():
    with st.spinner("Processing"):
        vectorstore_log = load_index_from_database(identifier="logistics")
        vectorstore_con = load_index_from_database(identifier="content")
        st.session_state.vectorstore_log = vectorstore_log
        st.session_state.vectorstore_con = vectorstore_con  
        #st.success("Documents processed successfully!")
