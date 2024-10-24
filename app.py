import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import traceback
import tempfile
import faiss

# Initialize Google API
def init_google_api():
    try:
        google_api_key = st.secrets["API_KEY"]
        if not google_api_key:
            st.error("API key not found in secrets.")
            return False
        genai.configure(api_key=google_api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Google API: {str(e)}")
        return False

# Modified PDF text extraction with better error handling
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            # Create a temporary file to handle the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file_path = tmp_file.name

            try:
                pdf_reader = PdfReader(tmp_file_path)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        if not text.strip():
            st.error("No text could be extracted from the PDF(s)")
            return None
        return text
    except Exception as e:
        st.error(f"Error in PDF extraction: {str(e)}")
        return None

# Modified text chunking with size verification
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Verify chunks
        if not chunks:
            st.error("Text splitting produced no chunks")
            return None
            
        st.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        st.error(f"Error in text chunking: {str(e)}")
        return None

# Modified vector store creation with persistent storage
def get_vector_store(text_chunks):
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            title="PDF Content"
        )

        # Create vector store with explicit dimension
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            normalize_L2=True  # Add L2 normalization
        )

        # Verify vector store
        if vector_store.index.ntotal == 0:
            st.error("Vector store creation failed - no vectors stored")
            return None

        st.success(f"Successfully created vector store with {vector_store.index.ntotal} vectors")
        return vector_store

    except Exception as e:
        st.error(f"Error in vector store creation: {str(e)}")
        traceback.print_exc()
        return None

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
        
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
        )
        
        return chain
    except Exception as e:
        st.error(f"Error in chain creation: {str(e)}")
        return None

def user_input(user_question, vector_store):
    try:
        if not vector_store:
            st.error("Vector store is not initialized")
            return

        # Add debug info
        st.info("Searching for relevant documents...")
        
        docs = vector_store.similarity_search(
            user_question,
            k=4,  # Number of relevant chunks to retrieve
            fetch_k=20  # Number of documents to initially fetch
        )
        
        st.info(f"Found {len(docs)} relevant documents")

        chain = get_conversational_chain()
        
        if chain:
            response = chain.invoke({
                "context": "\n\n".join([doc.page_content for doc in docs]),
                "question": user_question
            })
            
            st.markdown(f"<div style='font-size: 16px;'> 🤖 Response: {response}</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in processing question: {str(e)}")
        traceback.print_exc()

def main():
    st.set_page_config(page_title="📚 Chat PDF with Gemini AI", layout="centered", page_icon="📖")
    
    # Initialize API
    if not init_google_api():
        st.stop()
    
    st.markdown("<h1>Chat with Your PDF using Gemini AI 🤖</h1>", unsafe_allow_html=True)
    
    # Add session state initialization
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("📂 PDF Upload")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        st.info("Text extracted successfully")
                        
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            # Create vector store
                            vector_store = get_vector_store(text_chunks)
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.success("Ready to answer questions!")
            else:
                st.warning("Please upload PDF files first")

    # Main content area
    user_question = st.text_input("Ask a question about your PDFs:")
    
    if st.button("Get Answer"):
        if user_question:
            if st.session_state.vector_store is not None:
                user_input(user_question, st.session_state.vector_store)
            else:
                st.warning("Please process PDF files first")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()
