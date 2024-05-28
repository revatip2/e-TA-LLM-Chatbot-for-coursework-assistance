from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import mysql.connector
from config import sql_user, sql_pass, master_data_path, screenshots_path, pdf_slides_path, logistics_path, content_path, ocr_directory_path
import os
from utils import get_pdf_text
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def list_pdf_files(directory_path):
    files = os.listdir(directory_path)
    pdf_docs = [file for file in files if file.endswith('.pdf')]
    return pdf_docs


def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''


def get_text_chunks(text):
    text = str(text) if text is not None else ""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def serialize_faiss_index(vectorstore):
    pkl = vectorstore.serialize_to_bytes()  # serializes the faiss
    return pkl


def save_index_to_database(serialized_index, conn, identifier):
    
    cursor = conn.cursor()
    blob_data = serialized_index if isinstance(serialized_index, np.ndarray) else serialized_index
    print('Inserting..')
    cursor.execute("""
        INSERT INTO vector_stores (id, vector_store) VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE vector_store = %s
    """, (identifier, blob_data, blob_data))

    conn.commit()


# def process_logistics_data(logistics_path):
#     pdf_docs = list_pdf_files(logistics_path)
#     raw_text = get_pdf_text(pdf_docs, logistics_path)
#     text_chunks_pdf = get_text_chunks(raw_text)
#     print('Vector Stores..')
#     vector_store = get_vectorstore(text_chunks_pdf)
#     serialized_index = serialize_faiss_index(vector_store)
#     save_index_to_database(serialized_index, identifier="logistics")

# def process_content_data(content_path):
#     pdf_docs = list_pdf_files(content_path)
#     raw_text = get_pdf_text(pdf_docs, content_path)
#     text_chunks_pdf = get_text_chunks(raw_text)
#     data = pd.read_csv(master_data_path)
#     data['preprocessed_answer'] = data['Topic'] + " " + data['Answer']#.apply(preprocess_text)
#     texts = data['preprocessed_answer'].tolist()
#     print('Text Chunks..')
#     text_chunks = []
#     for text in texts:
#         chunks = get_text_chunks(text)
#         text_chunks.extend(chunks)
#     text_chunks.extend(text_chunks_pdf)
#     print('Vector Stores..')
#     vector_store = get_vectorstore(text_chunks)
#     serialized_index = serialize_faiss_index(vector_store)
#     save_index_to_database(serialized_index, identifier="content")

def process_logistics_data(logistics_path, conn):
    pdf_docs = list_pdf_files(logistics_path)
    raw_text = get_pdf_text(pdf_docs, logistics_path)
    text_chunks_pdf = get_text_chunks(raw_text)
    print('Vector Stores..')
    vector_store = get_vectorstore(text_chunks_pdf)
    serialized_index = serialize_faiss_index(vector_store)
    save_index_to_database(serialized_index, conn, identifier="logistics")

def process_content_data(content_path, conn):
    pdf_docs = list_pdf_files(content_path)
    raw_text = get_pdf_text(pdf_docs, content_path)
    text_chunks_pdf = get_text_chunks(raw_text)
    data = pd.read_csv(master_data_path)
    data['preprocessed_answer'] = data['Topic'] + " " + data['Answer']#.apply(preprocess_text)
    texts = data['preprocessed_answer'].tolist()
    print('Text Chunks..')
    text_chunks = []
    for text in texts:
        chunks = get_text_chunks(text)
        text_chunks.extend(chunks)
    text_chunks.extend(text_chunks_pdf)
    print('Vector Stores..')
    vector_store = get_vectorstore(text_chunks)
    serialized_index = serialize_faiss_index(vector_store)
    save_index_to_database(serialized_index, conn, identifier="content")

def insert_screenshots(screenshots_path, conn):
    screenshots_files = os.listdir(screenshots_path)
    screenshots_paths = [os.path.join(screenshots_path,file) for file in screenshots_files]

    for ss in screenshots_paths:
        if "temp_screenshot" in ss:
            continue
        video_title = ss[ss.index("Lecture"):ss.index("_screenshot_")]
        image_timestamp = ss[ss.index("_screenshot_")+12:-4]

        with open(ss, "rb") as file:
            image_data = file.read()
        query = "INSERT INTO screenshots (image_data, video_title, image_timestamp) VALUES (%s, %s, %s)"
        values = (image_data, video_title, image_timestamp)
        
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()


def insert_pdf_slides(pdf_slides_path, conn):
    slides_files = os.listdir(pdf_slides_path)
    slides_paths = [os.path.join(pdf_slides_path,file) for file in slides_files]

    for s in slides_paths:
        name = s[s.rindex("/")+1:]
        ppt_title = name.split("-")[1]
        slide_heading = name[name.index("_")+1:-4]

        with open(s, "rb") as file:
            image_data = file.read()
        query = "INSERT INTO slides (image_data, ppt_title, slide_heading) VALUES (%s, %s, %s)"
        values = (image_data, ppt_title, slide_heading)
        
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()



if __name__=="__main__":

    print('PDF Data..')
    pdf_docs = list_pdf_files(ocr_directory_path)
    raw_text = get_pdf_text(pdf_docs, ocr_directory_path)
    text_chunks_pdf = get_text_chunks(raw_text)

    print('Reading Data..')
    data = pd.read_csv(master_data_path)
    data['preprocessed_answer'] = data['Answer']#.apply(preprocess_text)

    texts = data['preprocessed_answer'].tolist()

    print('Text Chunks..')
    text_chunks = []
    for text in texts:
        chunks = get_text_chunks(text)
        text_chunks.extend(chunks)
    text_chunks.extend(text_chunks_pdf)

    print('Vector Stores..')
    vector_store = get_vectorstore(text_chunks)
    print(vector_store)

    def serialize_faiss_index(vectorstore):
        pkl = vectorstore.serialize_to_bytes()  # serializes the faiss
        return pkl

    serialized_index = serialize_faiss_index(vector_store)

    # process_logistics_data(logistics_path)
    # process_content_data(content_path)
    
    conn = mysql.connector.connect(
        host='localhost',
        user=sql_user,
        password=sql_pass,
        database='stars'
    )

    # print('Serialized..')
    # save_index_to_database(serialized_index, conn, identifier="unique_identifier_for_your_vector_store")

    print("Lecture Video Screenshots..")
    insert_screenshots(screenshots_path, conn)

    print("Lecture Slides PDF Images..")
    insert_pdf_slides(pdf_slides_path, conn)
    process_content_data(content_path, conn)
    process_logistics_data(logistics_path, conn)
    conn.close()
