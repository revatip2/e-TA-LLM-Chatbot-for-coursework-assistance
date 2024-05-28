# utils.py

# UNCOMMENT RESPONSE AND TIMESTAMP RESPONSE AND SCREENSHOTS 

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
import streamlit as st
from htmlTemplates import user_template, bot_template
from save_embeddings import serialize_faiss_index, save_index_to_database
from config import sql_user, sql_pass, llm_model_path, bert_model_path
import mysql.connector
from timestamps_urls import preprocess_transcripts, find_most_relevant_timestamp
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from images import get_screenshots, get_pdf_slides
from PIL import Image
from io import BytesIO


def get_pdf_text(pdf_docs, directory_path):
    text = ""
    for pdf_doc in pdf_docs:
        pdf = os.path.join(directory_path, pdf_doc)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("Vectorstore:",vectorstore)
    print("------------------------------------")
    return vectorstore

def get_conversation_chain(vectorstore):
    # model = "meta-llama/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model)
    
    # # Initialize the HuggingFace pipeline for text generation with Llama
    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     max_length=4096,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    # )
    
    # # Wrap the HuggingFace pipeline in your custom LLM wrapper
    # llm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={'temperature': 0})
    
    llm = LlamaCpp(
        #model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",  n_ctx=4096, n_batch=512)
        model_path=llm_model_path, n_ctx=8192, n_batch=512)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
    )

    return conversation_chain, llm

def get_query_similarity(user_query, vectorstore):
    """ Compute the cosine similarity between a user query and the k nearest embeddings in the vector store. """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode(user_query.lower(), convert_to_tensor=True)
    question_embedding = question_embedding.unsqueeze(0)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    result = retriever.get_relevant_documents(user_query)
 
    print(result)
    texts = [doc.page_content for doc in result]
    print(texts)

    # Vectorizing the documents and the user query using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [user_query])

    # Calculate cosine similarity between the user query and all documents
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Display the similarities
    for idx, doc in enumerate(result):
        print(f"Document {idx + 1}: {doc.page_content[:60]}... Similarity: {cosine_similarities[0][idx]}")

    return cosine_similarities

def should_add_to_embeddings(similarity_scores, vectorstore, user_question, response, identifier):

    threshold=0.15
    max_similarity = np.max(similarity_scores)
    print(f"Maximum Similarity Score: {max_similarity}")
    
    if max_similarity >= threshold:
        print("Adding the query and response to embeddings as they provide new insights.")
        data = user_question + " " + response['answer']
        texts = [data] 
        text_chunks = []
        for text in texts:
            chunks = get_text_chunks(text)
            text_chunks.extend(chunks)
        new_vectors = get_vectorstore(text_chunks)
        vectorstore.merge(new_vectors) 
        # update_embeddings(text_chunks, vectorstore)
        serialized_index = serialize_faiss_index(vectorstore)
        save_index_to_database(serialized_index, identifier)
        print('Updated embeddings.')

    else:
        print("Not adding to embeddings.")

def update_embeddings(new_text_chunks, vectorstore):
    """Update existing vector store with new tex    t chunks."""
    # Use the same model used to create the existing embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = embeddings.encode(new_text_chunks.lower(), convert_to_tensor=True)
    vs = vs.unsqueeze(0)
    vectorstore.add(vs)  # FAISS function to add new vectors
    serialized_index = serialize_faiss_index(vectorstore)
    save_index_to_database(serialized_index, "unique_identifier_for_your_vector_store")
    print('Updated embeddings.')

def classify_question(question, model, tokenizer, device='cpu'):
    model.to(device)
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs are on the right device
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()
    category = np.argmax(predictions.numpy())
    return category

def bert_mod(user_question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(bert_model_path , map_location='cpu'))
    model.eval()
    category = classify_question(user_question, model, tokenizer)
    return category

def perform_retrieval(user_question, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})
    relevant_passages = retriever.get_relevant_documents(user_question)
    print("relevant passages:",relevant_passages)
    print("------------------------------------")
    return relevant_passages


def generate_response(user_question, conversation_chain):
    # context = " ".join([passage.page_content for passage in relevant_passages])
    print("------------------------------------")
    print("User question:",user_question)
    print("------------------------------------")
    # print("Context:",context)
    # print("------------------------------------")
    if st.session_state.category == 0:
            full_prompt = f"Answer the question about the content of the course \"'{user_question}'\" using the following detailed context about the subject. Provide a detailed, accurate explanation. Tell the user that you cannot answer the question if the context does not help you."# Context: {context}"
    elif st.session_state.category == 1:
            full_prompt = f"Resolve the question about the logistics of the course \"'{user_question}'\" using the relevant information from the following context. Provide a clear and direct answer. Tell the user that you cannot answer the question if the context does not help you." #Context: {context}"

    # full_prompt = user_question + "\n" + "Consider the additional context provided to assist the user in \
    #     resolving the query with minimal need for additional clarification or information.\
    #          Emphasize accuracy and avoid generic statements that do not directly contribute to \
    #             the requirements of the query. If the context provided does not help you answer the question then clearly state that \
    #             you are unable to provide an accurate response.\
    #                 Context: " + context

    if len(full_prompt) > 2048:
        full_prompt = full_prompt[:2048]
    response = conversation_chain({'question': full_prompt})
    # response = {"answer":"RESPONSE HERE"}
    return response


def get_conversation_type():
    toggle = st.checkbox('Make question public?')
    return toggle



# def handle_userinput(user_question, vectorstore, conversation_chain):

#     if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
#         st.session_state.chat_history = []

#     relevant_passages = perform_retrieval(user_question, vectorstore)
#     print("Category is: ",st.session_state.category)
#     response = generate_response(user_question, relevant_passages, conversation_chain)

#     # response = {"answer":"SAMPLE RESPONSE"}

#     timestamps_df, model = preprocess_transcripts()
#     timestamp_response, ts = find_most_relevant_timestamp(timestamps_df,model, user_question)

#     print('Timestamp response: ',timestamp_response)

#     print("Response:",response)
#     print("------------------------------------")

#     # timestamp_response, ts = "https://www.youtube.com/watch?v=xoA5v9AO7S0&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV","0"
    
#     st.session_state.chat_history.append({'content': user_question, 'sender': 'user'})
#     st.session_state.chat_history.append({'content': timestamp_response, 'sender': 'bot_timestamp'})
#     st.session_state.chat_history.append({'content': response['answer'], 'sender': 'bot'})

#     conn = mysql.connector.connect(
#         host='localhost',
#         user=sql_user,
#         password=sql_pass,
#         database='stars'
#     )

#     # for i, message in enumerate(st.session_state.chat_history):
#     #     if message['sender'] == 'user':
#     #         st.subheader(message['content'])
#     #         # st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
#     #     elif message['sender'] == 'bot_timestamp':
#     #         st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
#     #         st.video(message['content'], start_time = int(ts))

#     with st.chat_message("assistant"):
#         st.subheader(user_question)
#         st.write(response['answer'])
#         st.video(timestamp_response, start_time = int(ts))
            
#         (ss1, ss2) = get_screenshots(timestamp_response, int(ts), conn)
#         ### first argument being passed is the link, the func defn takes video title -- CHANGE THE DEFN
#         if ss1:
#             st.write("Lecture Video Screenshots:")
#             img1 = Image.open(BytesIO(ss1[0]))
#             st.image(img1)
#         if ss2:
#             img2 = Image.open(BytesIO(ss2[0]))
#             st.image(img2)

#         pdf_img_id, pdf_img, pdf_title = get_pdf_slides(user_question, conn)
#         if pdf_img: 
#             st.write("Found in Lecture Slides:", pdf_title)
#             img = Image.open(BytesIO(pdf_img))
#             st.image(img)

#     conn.commit()
#     conn.close()

#     return response, timestamp_response, ts, pdf_img_id

def handle_userinput(user_question, vectorstore, conversation_chain):
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Retrieve relevant passages and generate a response
    # relevant_passages = perform_retrieval(user_question, vectorstore)
    response = generate_response(user_question, conversation_chain)
    st.session_state.chat_history.append({'content': user_question, 'sender': 'user'})
    st.session_state.chat_history.append({'content': response['answer'], 'sender': 'bot'})

    print("Category is: ", st.session_state.category)
    #print("Response: 0 ", response[0])

    if st.session_state.category == 0:  # Content question
        timestamps_df, model = preprocess_transcripts()
        timestamp_response, ts = find_most_relevant_timestamp(timestamps_df, model, user_question)
        st.session_state.chat_history.append({'content': timestamp_response, 'sender': 'bot_timestamp'})

        conn = mysql.connector.connect(
            host='localhost',
            user=sql_user,
            password=sql_pass,
            database='stars'
        )

        (ss1, ss2) = get_screenshots(timestamp_response, int(ts), conn)
        pdf_img_id, pdf_img, pdf_title = get_pdf_slides(user_question, conn)
        
        with st.chat_message("assistant"):
            st.subheader(user_question)
            st.write(response['answer'])
            if timestamp_response:
                st.video(timestamp_response, start_time=int(ts))
            if ss1:
                st.write("Lecture Video Screenshots:")
                img1 = Image.open(BytesIO(ss1[0]))
                st.image(img1)
            if ss2:
                img2 = Image.open(BytesIO(ss2[0]))
                st.image(img2)
            if pdf_img:
                st.write("Found in Lecture Slides:", pdf_title)
                img = Image.open(BytesIO(pdf_img))
                st.image(img)

        conn.close()
        return response, timestamp_response, ts, pdf_img_id

    else:  # Logistics question
        with st.chat_message("assistant"):
            st.subheader(user_question)
            st.write(response['answer'])
        # No additional information is processed or displayed
        return response, None, None, None
    

