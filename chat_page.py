# chat_page.py
import streamlit as st
from utils import handle_userinput, bert_mod, get_conversation_chain
from config import sql_user, sql_pass
import mysql.connector
from chat_history import save_chat_message

def save(user_question, response, video_link, ts, slide_id, is_public):
    conn = mysql.connector.connect(
        host='localhost',
        user=sql_user,
        password=sql_pass,
        database='stars'
    )
    save_chat_message(st.session_state.current_user, user_question, response, video_link, ts, slide_id, is_public, conn)
    conn.close()

def chat_page(vectorstore_log, vectorstore_con, conversation_chain):
        st.header("Chat with your TA Bot! :robot_face:")

        if 'response' not in st.session_state:
            st.session_state.response = None

        if 'category' not in st.session_state:
            st.session_state.category = None

        with st.form(key='my_form'):
            user_question = st.text_input("Ask your question about DSCI 553 Data Mining:", value="")
            ask_button = st.form_submit_button(label='Ask')

        if ask_button and user_question: 
                ques_category = bert_mod(user_question)
                print('Question Category: ', ques_category)
                st.session_state.category = ques_category

                vectorstore = vectorstore_con if ques_category == 0 else vectorstore_log

                # Check if we need to initialize or update the conversation chain
                if "conversation" not in st.session_state or "llm" not in st.session_state:
                    st.session_state.conversation, st.session_state.llm = get_conversation_chain(vectorstore)

                # Now process the input using the appropriate vector store and conversation chain
                response, video_link, ts, slide_id = handle_userinput(user_question, vectorstore, st.session_state.conversation)
                st.session_state.response = [response['answer'], video_link, ts, slide_id] 
            

                # if ques_category == 0:
                #     # if "conversation" not in st.session_state or "llm" not in st.session_state:
                #     #     st.session_state.conversation, st.session_state.llm = get_conversation_chain(st.session_state.vectorstore_con)
                #         response, video_link, ts, slide_id  = handle_userinput(user_question, vectorstore_con, conversation_chain)
                #         st.session_state.response = [response['answer'], video_link, ts, slide_id]  # Store response to use later
                # elif ques_category == 1:
                #     #  if "conversation" not in st.session_state or "llm" not in st.session_state:
                #     #     st.session_state.conversation, st.session_state.llm = get_conversation_chain(st.session_state.vectorstore_log)
                #         response, video_link, ts, slide_id  = handle_userinput(user_question, vectorstore_log, conversation_chain)
                #         st.session_state.response = [response['answer'], video_link, ts, slide_id]  # Store response to use later       
                    

        if st.session_state.response:
            if st.session_state.category == 0:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save in My Chats"): 
                        st.session_state.is_public = False
                        save(user_question, st.session_state.response[0], st.session_state.response[1], st.session_state.response[2], st.session_state.response[3], False)
                        st.session_state.response = None
                        st.rerun()
                        
                with col2:
                    if st.button("Save and Mark as Public"):  #green
                        st.session_state.is_public = True
                        save(user_question, st.session_state.response[0], st.session_state.response[1], st.session_state.response[2], st.session_state.response[3], True)
                        st.session_state.response = None
                        st.rerun()
            elif st.session_state.category == 1:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save in My Chats"): 
                        st.session_state.is_public = False
                        save(user_question, st.session_state.response[0], None, None, None, False)
                        st.session_state.response = None
                        st.rerun()
                        
                with col2:
                    if st.button("Save and Mark as Public"):  #green
                        st.session_state.is_public = True
                        save(user_question, st.session_state.response[0], None, None, None, True)
                        st.session_state.response = None
                        st.rerun()
                 
                


