import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Extract text from file
def get_text_from_file(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# Split text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store
def create_vector_store(text_chunks):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        embedding_results = embeddings.embed_documents(text_chunks)
        if not embedding_results:
            st.error("No embeddings generated.")
            return None

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store

    except Exception as e:
        st.error(f"Failed to create embeddings: {e}")
        return None

# Setup RAG pipeline
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say "Answer not available in the context".\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Main processing
def process_files(file):
    raw_text = get_text_from_file(file)
    if not raw_text.strip():
        st.error("The document has no readable text. Try another file.")
        return None
    text_chunks = get_chunks(raw_text)
    vector_store = create_vector_store(text_chunks)
    return vector_store

# Get answer
def get_answer(user_question, vector_store):
    if vector_store is None:
        return "Vector store is empty. Could not generate embeddings."
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Chat history
def add_to_history(user_query, bot_response):
    st.session_state.history.append({"role": "user", "message": user_query})
    st.session_state.history.append({"role": "bot", "message": bot_response})

# Streamlit UI setup
st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("📄 PDF & Word Document Q&A using Gemini")

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
if uploaded_file:
    with st.spinner("Processing your document..."):
        st.session_state.vector_store = process_files(uploaded_file)
        if st.session_state.vector_store:
            st.success(f"{uploaded_file.name} processed and embeddings generated.")
        else:
            st.warning("Could not generate vector store.")

# Question input
user_question = st.text_input("Ask a question about the document:")

# Answer button
if st.button("Get Answer"):
    if user_question:
        if st.session_state.vector_store:
            answer = get_answer(user_question, st.session_state.vector_store)
            add_to_history(user_question, answer)
        else:
            st.warning("Please upload and process a document first.")
    else:
        st.warning("Please enter a question.")

# Display chat history
if st.session_state.history:
    for message in reversed(st.session_state.history):
        if message["role"] == "user":
            st.markdown(
                f'<div style="text-align: right; background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin: 5px;">{message["message"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align: left; background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px;">{message["message"]}</div>',
                unsafe_allow_html=True
            )
