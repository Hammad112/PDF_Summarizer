import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import traceback

# Load the Google API key from Streamlit secrets
gemini_api_key = st.secrets.get("API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API key not found. Please check your .env file.")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# Cache the function that extracts text from PDF files
@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""
    return text

# Cache the function that splits text into chunks
@st.cache_data
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Cache the function that creates a vector store using FAISS
@st.cache_resource
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        traceback.print_exc()
        return None

# Set up the conversation chain with a prompt template
def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the provided context, respond with 
        "Answer is not available in the context."

        Context:
        {context}

        Question: 
        {question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        traceback.print_exc()
        return None

# Process user input to retrieve answer
def handle_user_input(user_question, vector_store):
    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.markdown(f"<div style='font-size: 16px;'> 🤖 Response: {response['output_text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing user question: {e}")
        traceback.print_exc()

# Main Streamlit app structure
def main():
    st.set_page_config(page_title="📚 Chat PDF with Gemini AI", layout="centered", page_icon="📖")

    # Styling and header
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #0A74DA;
        }
        .instruction {
            font-size: 18px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<h1 class='main-header'>Chat with Your PDF using Gemini AI 🤖</h1>", unsafe_allow_html=True)
    st.markdown("<p class='instruction'>Upload your PDF, ask questions, and get detailed AI responses!</p>", unsafe_allow_html=True)

    # Input for user questions
    user_question = st.text_input("🔍 Ask a Question from the PDF Files", placeholder="Type your question here...")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("📂 PDF Upload & Processing")
        st.write("1. Upload multiple PDFs.")
        st.write("2. Ask questions based on the content.")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

        # Process PDF files when user clicks "Submit & Process PDFs"
        if st.button("Submit & Process PDFs", key="pdf_process_button"):
            if pdf_docs:
                with st.spinner("📜 Extracting text and processing..."):
                    # Extract and process text, create vector store
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            vector_store = get_vector_store(text_chunks)
                            if vector_store:
                                st.session_state['vector_store'] = vector_store
                                st.success("✅ PDF processing complete!")
            else:
                st.warning("Please upload PDF files before processing.")

    # Process question if vector store exists and a question is provided
    if st.button("Submit Question", key="submit_question_button") and user_question:
        if 'vector_store' in st.session_state:
            handle_user_input(user_question, st.session_state['vector_store'])
        else:
            st.error("Please upload and process a PDF file first.")
    elif not user_question and st.button("Submit Question", key="submit_question_warning"):
        st.warning("Please enter a question before submitting.")

if __name__ == "__main__":
    main()

