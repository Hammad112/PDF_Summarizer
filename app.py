from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st

# Load the Google API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key not found. Please check your .env file.")

# Reading PDF's File content 
@st.cache_data
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

# Creating chunks of the Pdf's
@st.cache_data
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []
    return chunks

# Embeddings generation and Creating a vector store to save those embeddings
@st.cache_resource
def load_or_create_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("Text chunks are empty. Cannot create vector store.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Giving the prompt 
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
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        import traceback
        traceback.print_exc()
        return None
         
# loading prompt, chain, and user's prompt
def user_input(user_question, chain, vector_store):
    docs = vector_store.similarity_search(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response['output_text']


def main():
    # Set up Streamlit page and custom styles
    st.set_page_config(page_title=" Chat PDF with Gemini AI", layout="centered", page_icon="📖")
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

    # Page header and instructions
    st.markdown("<h1 class='main-header'>Chat with Your PDF 🤖</h1>", unsafe_allow_html=True)
    st.markdown("<p class='instruction'>Upload your PDF, ask questions, and get AI responses!</p>", unsafe_allow_html=True)

    # Set up session state for chat history and vector store
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar layout for PDF upload
    st.sidebar.title("📂 PDF Upload & Processing")
    st.sidebar.write("1. Upload multiple PDFs.")
    st.sidebar.write("2. Ask questions based on the content.")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

    # Initialize a flag for displaying success message and button
    processing_complete = False

    if pdf_docs and st.sidebar.button("Submit & Process PDFs"):
        with st.spinner("📜 Extracting text and processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            if not text_chunks:
                st.sidebar.error("No text extracted from the PDFs. Please check the content.")
            else:
                st.session_state.vector_store = load_or_create_vector_store(text_chunks)
                processing_complete = True  # Set flag to True on successful processing

    # Display success message and "Processed Successfully" button if processing is complete
    if processing_complete:
        st.sidebar.success("✅ Processing complete!")
      

    # Conversational chain setup
    chain = get_conversational_chain()

    # User question input and response display
    col1, col2 = st.columns([12, 2])
    with col1:
        user_question = st.text_input("🔍 Ask a Question from the PDF Files", placeholder="Type your question here...")
        if st.button("Submit"):
            if not user_question:
                st.warning("Please enter a question before submitting.")
            elif st.session_state.vector_store is None:
                st.warning("Please upload and process a PDF document before asking a question.")
            else:
                with st.spinner("🧠 Thinking..."):
                    answer = user_input(user_question, chain, st.session_state.vector_store)
                    st.session_state.chat_history.insert(0, (user_question, answer))  # Insert at the top

    # Display chat history with styling
    if st.session_state.chat_history:
        for question, answer in st.session_state.chat_history:
            st.write("---")
            st.markdown(f"**👤 You:** _{question}_")
            st.markdown(f"**🤖 AI:** {answer}")

if __name__ == "__main__":
    main()
