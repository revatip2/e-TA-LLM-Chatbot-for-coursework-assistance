# LLM Chatbot using Langchain, Llama2, RAGs
The e-TA is a **multi-turn, multi-modal LLM-powered chatbot** built to support the DSCI 553 - Data Mining course by automating student query responses. It leverages **Llama2**, **LangChain**, **Pinecone**, and **FAISS** to deliver real-time, personalized assistance, providing a multimodal learning experience with text, audio, and visual content.

Product Demo:  https://youtu.be/CUb7T9qn5Jw?si=80fXbvDCaQPRKlcf  

![image](https://github.com/user-attachments/assets/33865329-91ef-4a98-b8c8-1b627c79baa1)

## Key Features: BERT-Based Query Classification

- **BERT for Query Classification**: Utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to classify user queries into two categories:
  1. **Coursework-related**: Queries related to course material, concepts, and resources.
  2. **Logistics-related**: Queries related to administrative or scheduling information, such as deadlines or assignment details.

- **Coursework Pipeline**:
  - **Contextual Content Retrieval**: For coursework-related queries, BERT identifies relevant concepts, and the system returns **presentation slides** and **video segments** with **timestamps** related to the topic.
  - **Multimodal Response**: Supports both **text** and **video-based responses**, enhancing learning through comprehensive multimedia content delivery.

- **Logistics Pipeline**:
  - **Textual Response Generation**: For logistics-related queries (e.g., "When is the assignment due?"), BERT triggers a simpler pipeline that retrieves **text-based responses** without the need for multimedia content.

- **Prompt Engineering**: Custom **prompt engineering** ensures that the model distinguishes between different query types, guiding BERT to select the appropriate pipeline and content format based on the query classification.

- **Optimized Pipeline Setup**: The system is designed with dedicated pipelines for each query type:
  - **Coursework pipeline**: Focused on extracting and delivering rich, concept-based material.
  - **Logistics pipeline**: Streamlined for concise, informative text responses, minimizing processing overhead for simple queries.

## Methodology

1. Contextual Content Extraction
- Implements **Natural Language Processing (NLP)** techniques to extract key information from various sources, such as:
  - Lecture notes
  - Recordings
  - Assignments
  - Student discussion forums
- **Web scraping** and **API integrations** gather additional data from external public resources, enriching the e-TA’s knowledge base with real-world examples and trends in Data Mining.
- Utilizes **OpenAI Whisper** and **YouTube API** for **video transcriptions**, transcribing videos from external sources and linking relevant timestamps and content.
  
2. Generative AI & Retrieval-Augmented Generation (RAG)
- Leverages **Generative AI** powered by **Large Language Models (LLMs)**, specifically **Llama2**, to generate coherent, context-aware responses.
- **RAG** enhances the response generation process by retrieving relevant content from a **vector database** (using **FAISS**) before synthesizing responses in the LLM.
- **Advanced prompt engineering** ensures the system provides concise and accurate answers, with separate pipelines for **coursework**- and **logistics**-related queries.

3. Multimodal Learning
- Combines **text**, **audio**, and **visual** elements to offer a rich, immersive learning experience:
  - For coursework-related queries, the system retrieves **presentation slides** and **videos** with timestamps, providing a comprehensive response.

4. Feedback & Fine-tuning
- Continuously improves system performance by incorporating **student feedback**.


## How to Use

1. `pip install -r requirements.txt`
2. Download the llama 2 7B GGUF model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf and place it in the models folder
3. **Run the Streamlit App**
 `streamlit run app.py`
