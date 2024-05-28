# chat_page.py
import streamlit as st
from utils import handle_userinput
from chat_history import get_public_chat_history, show_chat_history
from config import sql_user, sql_pass
import mysql.connector



def public_page():
    st.header("Public Forum :robot_face:")

    conn = mysql.connector.connect(
                host='localhost',
                user=sql_user,
                password=sql_pass,
                database='stars'
            )

    chat_history = get_public_chat_history(conn)
    show_chat_history(chat_history)

    conn.close()

  