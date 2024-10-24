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
from time import time

# Add timeout decorator
def timeout_handler(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        if time() - start_time > 60:  # 60 seconds timeout
            st.error("Processing timed out. Please try with a smaller PDF file.")
            return None
        return result
    return wrapper

# Modified PDF text extraction with progress bar
@timeout_handler
def get_pdf_text(pdf_docs):
    text = ""
    try:
        total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
        progress_bar = st.progress(0)
        current_page = 0
        
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
                current_page += 1
                progress_bar.progress(current_page / total_pages)
                
        progress_bar.empty()
        if not text.strip():
            st.warning("No text could be extracted from the PDF(s). Please check if the PDFs contain searchable text.")
            return None
    except Exception as e:
        st.error(f"Error reading PDF files: {str(e)}")
        return None
    return text

# Modified text chunking with size limit
def get_text_chunks(text):
    try:
        if len(text) > 1000000:  # Add size limit
            st.warning("Text content is very large. Processing may take longer.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Reduced chunk size
            chunk_overlap=500,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return None

# Modified vector store creation with progress indication
def get_vector_store(text_chunks):
    try:
        with st.spinner("Creating vector embeddings..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="📚 Chat PDF with Gemini AI", layout="centered", page_icon="📖")
    
    # Add CSS styling (same as before)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>Chat with Your PDF using Gemini AI 🤖</h1>", unsafe_allow_html=True)
    
    # Modified sidebar layout
    with st.sidebar:
        st.title("📂 PDF Upload & Processing")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process PDFs"):
            if pdf_docs:
                # Check file sizes
                total_size = sum(pdf.size for pdf in pdf_docs)
                if total_size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("Total PDF size too large. Please upload smaller files.")
                    return
                
                # Process PDFs with status updates
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.warning("Please upload PDF files first.")

    # Main content area
    user_question = st.text_input("🔍 Ask a Question:", placeholder="Type your question here...")
    
    if st.button("Get Answer"):
        if user_question:
            if 'vector_store' in st.session_state:
                try:
                    with st.spinner("Searching for answer..."):
                        user_input(user_question, st.session_state.vector_store)
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
            else:
                st.warning("Please process PDF files first.")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
