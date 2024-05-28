from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from config import timestamps_data_path

def preprocess_transcripts():
    df = pd.read_csv(timestamps_data_path)
    df['processed_text'] = df['text'].str.lower().str.replace('[^\w\s]', '', regex=True)
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings for the processed text
    df['embeddings'] = df['processed_text'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df, model


def find_most_relevant_timestamp(df, model, question):

    # Encode the user's question
    question_embedding = model.encode(question.lower(), convert_to_tensor=True)
    
    # Ensure question_embedding is 2D with shape (1, embedding_size)
    question_embedding = question_embedding.unsqueeze(0)
    
    # Calculate cosine similarity with all transcript embeddings
    similarities = df['embeddings'].apply(lambda x: torch.nn.functional.cosine_similarity(question_embedding, x.unsqueeze(0)).item())
    
    # Find the index of the most similar embedding
    most_relevant_idx = similarities.idxmax()
    
    # Return the timestamp and URL of the most relevant entry
    relevant_info = df.loc[most_relevant_idx, ['start', 'url']]
    print(relevant_info)
    ts = relevant_info['start']
    url = relevant_info['url']
    url_video = f'{url}&t={ts}s'
    return url_video, ts
