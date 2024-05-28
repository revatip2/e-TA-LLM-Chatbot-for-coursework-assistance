import streamlit as st
import mysql.connector
import hashlib
from config import sql_user, sql_pass

def home():
    st.header("DSCI 553 - Data Mining | Chat with your eTA")

# Function to hash the password using SHA-256
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Establishing a connection to MySQL database
def connect_to_database():


    return mysql.connector.connect(
        host='localhost',
        user=sql_user,
        password=sql_pass,
        database='stars'
    )

def create_user(username, password):
    """Create a new user."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
    if cursor.fetchone():
        conn.close()
        return False
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hash_password(password)))
    conn.commit()
    conn.close()
    return True

def check_user(username, password):
    """Check if the user exists and the password is correct."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    if user and user[1] == hash_password(password):
        return True
    return False


def signup_login_page():
    """Display the signup/login page."""

    if st.session_state.current_user:
        return
    home()

    choice_placeholder = st.empty()
    menu = ["Login", "Signup"]
    choice = choice_placeholder.radio("Login / Signup", menu)
    
    title_placeholder = st.empty()  
    username_placeholder = st.empty()  
    password_placeholder = st.empty() 
    button_placeholder = st.empty() 

    if choice == "Signup":
        title_placeholder.title("Signup")  
        username = username_placeholder.text_input("Username")
        password = password_placeholder.text_input("Password", type="password")
        if button_placeholder.button("Signup"):
            if create_user(username, password):
                st.success("Signup successful! You can now login.")
                title_placeholder.empty()  
                username_placeholder.empty()  
                password_placeholder.empty()  
                button_placeholder.empty()
            else:
                st.error("Username already exists")
            

    elif choice == "Login":
        title_placeholder.title("Login")  
        username = username_placeholder.text_input("Username")
        password = password_placeholder.text_input("Password", type="password")
        if button_placeholder.button("Login"):
            if check_user(username, password):
                st.session_state.current_user = username  # Store current user
                st.success(f"Logged in as {username}")
                choice_placeholder.empty()
                title_placeholder.empty()  
                username_placeholder.empty()
                password_placeholder.empty()
                button_placeholder.empty()
            else:
                st.error("Invalid username or password.")

if __name__ == "__main__":
    signup_login_page()