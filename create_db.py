import mysql.connector
from config import sql_user, sql_pass
import os
            
conn = mysql.connector.connect(
    host='localhost',
    user=sql_user,
    password=sql_pass,
    database='stars'
)

cursor = conn.cursor()

# cursor.execute('''
#     CREATE DATABASE IF NOT EXISTS stars;
# '''
# )

cursor.execute('''
    USE stars;
'''
)

cursor.execute('''
    DROP TABLE IF EXISTS slides;
''')

cursor.execute('''
    DROP TABLE IF EXISTS screenshots;
''')

cursor.execute('''
    DROP TABLE IF EXISTS chat_history;
''')

cursor.execute('''
    DROP TABLE IF EXISTS users;
'''
)                          

cursor.execute('''
    DROP TABLE IF EXISTS vector_stores;
''')
            
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username VARCHAR (20),
    password VARCHAR (256),
    PRIMARY KEY (username));
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS vector_stores (
    id VARCHAR(255) NOT NULL,
    vector_store LONGBLOB,
    PRIMARY KEY (id));
''')

# cursor.execute('''
# CREATE TABLE IF NOT EXISTS chat_history (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     username VARCHAR(20),
#     message TEXT,
#     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     is_user_message BOOLEAN,
#     video_link VARCHAR(255),
#     ts INT,
#     slide_id INT,
#     is_public BOOLEAN,
#     FOREIGN KEY (username) REFERENCES users(username)
# );
# ''')


cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT,
    text_response TEXT,
    video_link VARCHAR(255) DEFAULT NULL,
    video_ts INT DEFAULT NULL,
    slide_id INT DEFAULT NULL,
    is_public BOOLEAN,
    FOREIGN KEY (username) REFERENCES users(username)
);
''')


cursor.execute('''
CREATE TABLE IF NOT EXISTS screenshots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_data LONGBLOB,
    video_title VARCHAR(255),
    image_timestamp FLOAT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS slides (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_data LONGBLOB,
    ppt_title VARCHAR(255),
    slide_heading VARCHAR(255)
);
''')

conn.commit()
conn.close()
