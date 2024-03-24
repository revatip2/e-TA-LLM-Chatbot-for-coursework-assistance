# PDFs Chatbot using Langchain, Llama2
This project allows you to upload PDF documents, process them, and then engage in a conversational interface where you can ask questions about the documents.

## Getting Started

### Install Dependencies

`pip install -r requirements.txt`

1. Download the llama 2 7B GGUF model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf and place it in the models folder

### How it works 
1. The application gui is built using streamlit
2. The application reads text from PDF files, splits it into chunks
3. Uses HuggingFaceEmbeddings to generate embedding vectors used to find the most relevant content to a user's question 
4. Build a conversational retrieval chain using Langchain
5. Use Llama2 to generate response based on content in PDF

### Code Structure

The code is structured as follows:

- **utils.py**
    - Contains utility functions for PDF text extraction and chunking.
    - Utilizes HuggingFace Transformers to generate embedding vectors from text chunks and creates a vector store using Faiss.
    - Builds a conversational retrieval chain using Langchain with LlamaCpp model for context generation.

- **app.py**
    - The main Streamlit application file.
    - It handles navigation between the upload/process and chat interfaces.
    - Based on the user's selection from the sidebar menu, the application routes the user to either the "Upload & Process Documents" page or the "Chat" page.

- **upload_page.py**
    - Implementation for the upload and processing page.
    - Allows users to upload PDFs and process them.

- **chat_page.py**
    - Implementation for the chat interface.
    - Enables users to interact with the system by asking questions.
      
- **htmlTemplates.py**
    - Contains HTML templates for displaying user and bot messages.

## How to Use

1. **Run the Streamlit App**
 `streamlit run app.py`
2. **Upload & Process Documents**
    - This GUI allows the user to upload PDF documents and process them to extract text and create a vector store for conversational retrieval.
3. **Chat**
    - Once the documents are processed, there is a different GUI which now allows the user to start asking questions related to the uploaded documents.

Users can switch between the two functionalities using the dropdown menu in the application.
