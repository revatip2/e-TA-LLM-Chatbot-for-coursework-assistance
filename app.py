# app.py
import streamlit as st
from auth import signup_login_page, check_user
from upload_page import upload_page
from chat_page import chat_page
from public_page import public_page
from utils import get_conversation_chain
from config import sql_pass, sql_user
from htmlTemplates import bot_template, user_template
import mysql.connector
from chat_history import get_user_chat_history, show_chat_history


def main():
    upload_page()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore_log" not in st.session_state:
        st.session_state.vectorstore_log = None
    if "vectorstore_con" not in st.session_state:
        st.session_state.vectorstore_con =None
    if 'category' not in st.session_state:
            st.session_state.category = None
               
    print('Session State: ', st.session_state)
    if 'current_user' in st.session_state:
        print('Current User in Session state..', st.session_state.current_user)

        tab1, tab2 = st.tabs(["Ask a Question", "View Public Questions"])

        with tab1:
            chat_page(st.session_state.vectorstore_log, st.session_state.vectorstore_con, st.session_state.conversation)

            conn = mysql.connector.connect(
                host='localhost',
                user=sql_user,
                password=sql_pass,
                database='stars'
            )
            
            st.markdown("-------------------------------------------------------")
            st.header("My Chats")
            chat_history = get_user_chat_history(st.session_state.current_user, conn)
            show_chat_history(chat_history)
                
            conn.commit()
            conn.close()
            
        with tab2:
            public_page() 

        if st.sidebar.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]  # Clear all session state
            st.rerun()  # Rerun the app to reflect changes


        
    if st.session_state.vectorstore_log is None or st.session_state.vectorstore_con is None:
        st.warning("No documents found.")
    # else:
    #     if "conversation" not in st.session_state or "llm" not in st.session_state:
    #         # print("Category session state is: ", st.session_state.category)
    #         # if st.session_state.category == 0:
    #         st.session_state.conversation, st.session_state.llm = get_conversation_chain(st.session_state.vectorstore_con)
    #         # elif st.session_state.category == 1:
    #         #     st.session_state.conversation, st.session_state.llm = get_conversation_chain(st.session_state.vectorstore_log)
        
if __name__ == '__main__':
    st.set_page_config(page_title="DSCI 553 TA Bot", page_icon=":robot_face:")

    if "current_user" in st.session_state:
        print("Logged in as ",st.session_state.current_user)

    # Call the authentication page if the user is not logged in
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
        
    if st.session_state.current_user is None:
        signup_login_page()
    
    if st.session_state.current_user:
        main()  # Show the main app if the user is logged in

