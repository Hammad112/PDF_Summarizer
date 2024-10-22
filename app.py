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

# Ensure the Google API key is loaded
google_api_key = st.secrets.get("API_KEY")

# Check if the API key is set, otherwise show an error
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("API key not found. Please check your environment variables or Streamlit secrets.")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []
    return chunks

# Function to create an in-memory FAISS vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        traceback.print_exc()
        return None

# Function to create a conversation chain with Google Generative AI
def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, say, "Answer is not available in the context." Do not provide a wrong answer.

        Context:
        {context}

        Question: 
        {question}

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        traceback.print_exc()
        return None

# Function to process user input and provide a response
def user_input(user_question, vector_store):
    try:
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.markdown(f"<div style='font-size: 16px;'> 🤖 Response:: {response['output_text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing user input: {e}")
        traceback.print_exc()

# Main function to handle Streamlit UI and actions
def main():
    # Set page title and icon
    st.set_page_config(page_title="📚 Chat PDF with Gemini AI", layout="centered", page_icon="📖")
    
    # Add CSS for styling
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

    # Add header
    st.markdown("<h1 class='main-header'>Chat with Your PDF using Gemini AI 🤖</h1>", unsafe_allow_html=True)
    st.markdown("<p class='instruction'>Upload your PDF, ask questions, and get detailed AI responses!</p>", unsafe_allow_html=True)

    # Create a 2-column layout for better structure
    col1, col2 = st.columns([12, 2])

    with col1:
        user_question = st.text_input("🔍 Ask a Question from the PDF Files", placeholder="Type your question here...")

        # Add a "Submit" button to process the question
        if st.button("Submit"):
            if user_question:
                st.write("### 🧠 Thinking...")
                # Only allow submission if vector_store is available
                if 'vector_store' in st.session_state:
                    user_input(user_question, st.session_state.vector_store)
                else:
                    st.error("Please upload and process a PDF file first.")
            else:
                st.warning("Please enter a question before submitting.")

    with col2:
        with st.sidebar:
            st.title("📂 PDF Upload & Processing")
            st.write("1. Upload multiple PDFs.")
            st.write("2. Ask questions based on the content.")
            pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

            if st.button("Submit & Process PDFs"):
                if pdf_docs:
                    with st.spinner("📜 Extracting text and processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks:
                                vector_store = get_vector_store(text_chunks)
                                if vector_store:
                                    # Store vector store in session state to avoid re-processing
                                    st.session_state.vector_store = vector_store
                                    st.success("✅ Processing complete!")
                else:
                    st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
