import streamlit as st
import os
import numpy as np
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document

# Set page configuration
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Application title
st.title("📚 Document Chat Assistant")
st.markdown("Upload documents and ask questions about them!")

# Sidebar for uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, DOCXs, or TXT files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    process_button = st.button("Process Documents")

# Initialize session state variables if they don't exist
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'document_vectors' not in st.session_state:
    st.session_state.document_vectors = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to load and process documents
def process_documents(files):
    # Create a temporary directory to save uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each uploaded file
        documents = []
        for file in files:
            temp_filepath = os.path.join(temp_dir, file.name)
            
            # Save the uploaded file to disk
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            
            # Load documents based on file type
            try:
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_filepath)
                    documents.extend(loader.load())
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(temp_filepath)
                    documents.extend(loader.load())
                elif file.name.endswith(".txt"):
                    loader = TextLoader(temp_filepath)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks

# Custom retrieval function using TF-IDF
def retrieve_most_similar(query, documents, vectorizer, document_vectors, k=3):
    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity scores
    similarity_scores = (query_vector @ document_vectors.T).toarray()[0]
    
    # Get the top k most similar document indices
    top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
    
    # Return the top k documents
    return [documents[i] for i in top_k_indices]

# Process documents when the button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        # Process documents
        chunks = process_documents(uploaded_files)
        
        if chunks:
            try:
                with st.spinner("Creating document index (this may take a few moments)..."):
                    # Extract text from chunks
                    texts = [doc.page_content for doc in chunks]
                    
                    # Create TF-IDF vectorizer
                    vectorizer = TfidfVectorizer()
                    document_vectors = vectorizer.fit_transform(texts)
                    
                    # Save to session state
                    st.session_state.documents = chunks
                    st.session_state.vectorizer = vectorizer
                    st.session_state.document_vectors = document_vectors
                    st.session_state.processed = True
                    
                    st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents into {len(chunks)} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error creating index: {str(e)}")
        else:
            st.sidebar.error("No content extracted from documents.")
elif process_button and not uploaded_files:
    st.sidebar.error("Please upload at least one document.")

# Main chat interface
if st.session_state.processed and st.session_state.documents:
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = retrieve_most_similar(
                        user_question,
                        st.session_state.documents,
                        st.session_state.vectorizer,
                        st.session_state.document_vectors,
                        k=3
                    )
                    
                    # Display relevant context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Generate a simple response
                    answer = f"Based on the documents, I found the following relevant information:\n\n{context}"
                    
                    # Display the answer
                    st.write(answer)
                    
                    # Display source information in an expander
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                except Exception as e:
                    error_message = f"Error searching documents: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
else:
    # Display instructions when no documents are processed yet
    if not uploaded_files:
        st.info("👈 Please upload your documents in the sidebar and click 'Process Documents'.")
    else:
        st.info("👈 Click 'Process Documents' to start.")

# Add information about the application
with st.expander("About this App"):
    st.markdown("""
    ### How it works
    This app uses TF-IDF (Term Frequency-Inverse Document Frequency) to create a simple document retrieval system:
    
    1. Upload PDF, DOCX, or TXT documents using the sidebar
    2. Click "Process Documents" to analyze your files
    3. Ask questions about your documents in the chat interface
    
    ### Technologies Used
    - **Streamlit**: Web application framework
    - **TF-IDF**: Text vectorization technique for document similarity
    - **LangChain**: Document loading and processing
    
    ### Required Packages
    ```
    pip install streamlit langchain langchain-community langchain-text-splitters scikit-learn pypdf python-docx docx2txt
    ```
    
    ### Features and Limitations
    - This version uses TF-IDF for document similarity rather than a language model
    - It retrieves and displays the most relevant text passages from your documents
    - The app does not require any API keys or external services
    - All processing happens locally in your browser
    
    ### Privacy Note
    Your documents are processed temporarily and not stored permanently.
    """)
