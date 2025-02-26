import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))

index_name = "doc-chat-db"

if index_name not in [i['name'] for i in pc.list_indexes()] :
    pc.create_index(
        name=index_name,
        dimension=768,
        metric = 'cosine',
        spec = ServerlessSpec(cloud = 'aws', region = 'us-east-1'),
    )

index = pc.Index(index_name)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = PineconeVectorStore(index = index, embedding=embeddings)
    vector_store.add_texts(text_chunks)
    return vector_store

def get_rel_text(user_question, db):
    docs = db.similarity_search(user_question, k = 1)
    return docs[0].page_content

def bot_response(model, query, relevant_texts, history): 
    context = ' '.join(relevant_texts)
    prompt = f"""This is the context of the document 
    Context: {relevant_texts}
    And this is the user query
    User: {query}
    And this is the history of the conversation
    History: {history}

    Please generate a response to the user query based on the context and history
    The questions might be asked related to the provided context, and may also be in terms of the external content related to the document,
    Answer the query with respect to the context provided, you can also use your additional knowledge too, but do not ignore the content of the provided context,
    Answer the queries like a professional person being in the domain of the context provided, having a lot of knowledge on the based report context
    Bot:
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.65,
        )
    )
    return response.text

st.set_page_config(page_title = "Document Chat", layout = "wide")
st.title("Document Information Retriever")

if "doc_paragraphs" not in st.session_state:
    st.session_state.doc_paragraphs = {}
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = {}
if "faiss" not in st.session_state:
    st.session_state.faiss = {}

st.sidebar.subheader("File Uploading")

slt = st.sidebar.selectbox(label="Choose the type of document chat", options = ["Single doc chat", "Multiple doc chat"])

if slt == "Single doc chat":
    s_file = st.sidebar.file_uploader("Upload your PDF files", type = ['pdf'], accept_multiple_files = False)

    if s_file:
        if st.sidebar.button("Upload file.."):
            if s_file.file_id not in st.session_state.doc_messages:
                st.session_state.doc_messages[s_file.file_id] = []

            if s_file.file_id not in st.session_state.doc_paragraphs:
                with st.spinner('Getting the details'): 
                    pdf_reader = PdfReader(s_file)
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    st.session_state.doc_paragraphs[s_file.file_id] = text

            if s_file.file_id not in st.session_state.faiss:
                chunks = get_chunks(st.session_state.doc_paragraphs[s_file.file_id])

                with st.spinner("Reading records..."):
                    st.session_state.faiss[s_file.file_id] = get_vector_store(chunks)

            if s_file.file_id in st.session_state.faiss:
                st.info("File is uploaded, start the chat.. !!")

        h_model = genai.GenerativeModel(model_name= "gemini-2.0-flash", 
        system_instruction = "You are a very professional person related to any domain, and can answer any queries, related to the document in an easier manner"
        )

        doc_chat = st.session_state.doc_messages.get(s_file.file_id, [])

        for message in doc_chat:
            row = st.columns(2)
            if message['role'] == 'user':
                row[1].chat_message(message['role']).markdown(message['content'])
            else:
                row[0].chat_message(message['role']).markdown(message['content'])

        try:
            user_question = st.chat_input("Enter your query here !!")

            if user_question:
                row_u = st.columns(2)
                row_u[1].chat_message('user').markdown(user_question)
                doc_chat.append(
                    {'role': 'user',
                    'content': user_question}
                )

                with st.spinner("Generating response..."):
                    relevant_texts = get_rel_text(user_question, st.session_state.faiss[s_file.file_id])
                    bot_reply = bot_response(h_model, user_question, relevant_texts, doc_chat)

                row_a = st.columns(2)
                row_a[0].chat_message('assistant').markdown(bot_reply)

                doc_chat.append(
                    {'role': 'assistant',
                    'content': bot_reply}
                )

        except Exception as e:
            st.chat_message('assistant').markdown(f'There might be an error, try again, {str(e)}')
            doc_chat.append(
                {
                    'role': 'assistant',
                    'content': f'There might be an error, try again, {str(e)}'
                }
            )

    else:
        st.warning("Please upload any documents to start the chat...")

else:
    s_files = st.sidebar.file_uploader("Upload your PDF files",help = "You should be uploading more than one file", type = ['pdf'], accept_multiple_files = True)
    s_files_id = ""
    for s_file in s_files:
        s_files_id += s_file.file_id
    
    if len(s_files) >= 2:
        if st.sidebar.button("Upload file"):
            try:
                if s_files_id not in st.session_state.doc_messages:
                    st.session_state.doc_messages[s_files_id] = []
                
                texts = ""
                for s_file in s_files:
                    if s_file.file_id not in st.session_state.doc_paragraphs:
                        with st.spinner('Getting the details'):
                            pdf_reader = PdfReader(s_file)
                            text = ''
                            for page in pdf_reader.pages:
                                text+= page.extract_text()
                    
                            st.session_state.doc_paragraphs[s_file.file_id] = text

                    texts+=st.session_state.doc_paragraphs[s_file.file_id]
                
                st.session_state.doc_paragraphs[s_files_id] = texts
                    
                if s_files_id not in st.session_state.faiss:
                    chunks = get_chunks(st.session_state.doc_paragraphs[s_files_id])

                    with st.spinner("Reading records..."):
                        st.session_state.faiss[s_files_id] = get_vector_store(chunks)

                if s_files_id in st.session_state.faiss:
                    st.info("The files are uploaded, you can start the chat now...")
            
            except Exception as e:
                st.error(f"Error Occurred: {e}")

        h_model = genai.GenerativeModel(model_name= "gemini-2.0-flash", 
        system_instruction = "You are a very professional person related to any domain, and can answer any queries, related to the document in an easier manner"
        )

        doc_chat = st.session_state.doc_messages.get(s_files_id, [])

        for message in doc_chat:
            row = st.columns(2)
            if message['role'] == 'user':
                row[1].chat_message(message['role']).markdown(message['content'])
            else:
                row[0].chat_message(message['role']).markdown(message['content'])

        try:
            user_question = st.chat_input("Enter your query here !!")

            if user_question:
                row_u = st.columns(2)
                row_u[1].chat_message('user').markdown(user_question)
                doc_chat.append(
                    {'role': 'user',
                    'content': user_question}
                )

                with st.spinner("Generating response..."):
                    relevant_texts = get_rel_text(user_question, st.session_state.faiss[s_files_id])
                    bot_reply = bot_response(h_model, user_question, relevant_texts, doc_chat)

                row_a = st.columns(2)
                row_a[0].chat_message('assistant').markdown(bot_reply)

                doc_chat.append(
                    {'role': 'assistant',
                    'content': bot_reply}
                )

        except Exception as e:
            st.chat_message('assistant').markdown(f'There might be an error, try again, {str(e)}')
            doc_chat.append(
                {
                    'role': 'assistant',
                    'content': f'There might be an error, try again, {str(e)}'
                }
            )

    elif len(s_files) == 1:
        st.warning("Please upload more than one document to start the chat..")

    else:
        st.warning("Please upload the documents to start the chat...")

