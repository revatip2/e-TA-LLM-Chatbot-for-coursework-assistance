from PIL import Image
from io import BytesIO
import mysql.connector
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_screenshots(video_title, timestamp, conn):
    cursor = conn.cursor()

    query1 = ''' 
    SELECT image_data
    FROM screenshots
    WHERE video_title LIKE %s AND image_timestamp <= %s
    ORDER BY image_timestamp DESC
    LIMIT 1
    '''
    cursor.execute(query1, ('%'+video_title+'%', float(timestamp)))
    img1 = cursor.fetchone()

    query2 = ''' 
    SELECT image_data
    FROM screenshots
    WHERE video_title LIKE %s AND image_timestamp >= %s
    ORDER BY image_timestamp DESC
    LIMIT 1
    '''
    cursor.execute(query2, ('%'+video_title+'%', float(timestamp)))
    img2 = cursor.fetchone()

    # Debug statements
    print("VIDEO TITLE, TIMESTAMP", video_title, timestamp)
    print("TYPE OF IMAGE 1",type(img1))
    # print("IMAGE 1")
    # print(img1)

    return (img1, img2)


def get_pdf_slides(user_query, conn):
    print("inside pdf slides")

    # Initialize tokenizer and model from Hugging Face
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def embed_text(text):
        # Tokenize and get embeddings
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get sentence embeddings
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # Move to CPU and convert to numpy array
        return mean_embeddings.cpu().numpy()

    cursor = conn.cursor()

    # Fetch data from MySQL
    query = "SELECT id, ppt_title, slide_heading FROM slides"
    cursor.execute(query)
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['id', 'ppt_title', 'slide_heading'])
    df['combined_text'] = df['ppt_title'] + " " + df['slide_heading']

    # user_query = "how do i solve this k means problem"

    # Embedding all texts including the user query
    embeddings = [embed_text(text) for text in df['combined_text'].tolist() + [user_query]]
    embedding_matrix = np.vstack(embeddings)

    # Compute cosine similarity
    cos_sim = cosine_similarity(embedding_matrix[-1].reshape(1, -1), embedding_matrix[:-1]).flatten()

    # Find indices of the three most similar records
    most_similar_indices = cos_sim.argsort()[-2:][::-1]

    # Fetch and display the most similar records along with their similarity scores
    for idx in most_similar_indices:
        similar_record = df.iloc[idx]
        similarity_score = cos_sim[idx]
        print(similar_record['id'])
        print(f"Record: {similar_record['ppt_title']} - {similar_record['slide_heading']}")
        print(f"Similarity Score: {similarity_score}\n")

    top_idx = df.iloc[most_similar_indices[0]]['id']

    if cos_sim[most_similar_indices[0]] > 0.3:
        query = ''' 
        SELECT id, image_data, ppt_title
        FROM slides
        WHERE id = %s
        '''
        cursor.execute(query,(int(top_idx),))
        pdf_img_id, pdf_img, pdf_title = cursor.fetchone()

        return pdf_img_id, pdf_img, pdf_title
    
    return None, None, None
