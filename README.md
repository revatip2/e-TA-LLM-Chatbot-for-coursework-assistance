# Mulitmodal LLM Chatbot using Langchain, Llama2, RAGs
This project allows you to upload PDF documents, process them, and then engage in a conversational interface where you can ask questions about the documents.

## Getting Started

### Install Dependencies

`pip install -r requirements.txt`

1. Download the llama 2 7B GGUF model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf and place it in the models folder

### How it works 
1. The application gui is built using streamlit
2. The application processes vudeo transcripts, images, timestamp data, and text files 
3. Uses HuggingFaceEmbeddings to generate embedding vectors used to find the most relevant content to a user's question 
4. Build a conversational retrieval chain using Langchain and employed RAGs
5. A query classifier classifies user question into content versus logistics and a separate pipeline is followed based on classification
6. Use Llama2 to generate response and return an answer with video timestamp, and images embedded for content type query

## How to Use

1. **Run the Streamlit App**
 `streamlit run app.py`
