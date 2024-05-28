import streamlit as st
from htmlTemplates import user_template, bot_template
from PIL import Image
from io import BytesIO


def save_chat_message(username, question, text_response, video_link, video_ts, slide_id, is_public, conn):
    print(f'Saving Chat History for {username}')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO chat_history (username, question, text_response, video_link, video_ts, slide_id, is_public) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ''', (username, question, text_response, video_link, video_ts, slide_id, is_public))
    print('Done!')
    conn.commit()


def get_user_chat_history(username, conn):
    print(f'Retrieving Chat History for {username}')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT ch.timestamp, ch.question, ch.text_response, ch.video_link, ch.video_ts, s.ppt_title, s.image_data FROM chat_history ch
    JOIN slides s
    ON ch.slide_id = s.id
    WHERE username = %s 
    ORDER BY timestamp DESC
    ''', (username,))
    print('Done!')
    return cursor.fetchall()


def get_public_chat_history(conn):
    print(f'Retrieving Public Chat History')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT ch.timestamp, ch.question, ch.text_response, ch.video_link, ch.video_ts, s.ppt_title, s.image_data FROM chat_history ch
    JOIN slides s
    ON ch.slide_id = s.id
    WHERE is_public = 1
    ORDER BY timestamp DESC               
    ''')
    print('Done!')
    return cursor.fetchall()


def show_chat_history(chat_history):
        # st.header("Chat History")
        # st.markdown("-------------------------------------------------------")
        for timestamp, question, text_response, video_link, video_ts, ppt_title, slide_image in chat_history:

            st.write(timestamp)
            with st.expander(question, expanded=False):  
                st.write(text_response)
                if video_link:
                    st.write("Found in Lecture Videos:", ppt_title)
                    st.video(video_link, start_time = int(video_ts))
                if ppt_title:
                    st.write("Found in Lecture Slides:", ppt_title)
                    img = Image.open(BytesIO(slide_image))
                    st.image(img)
            

    