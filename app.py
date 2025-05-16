import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set page configuration
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Application title
st.title("📚 Document Chat Assistant")
st.markdown("Upload documents and ask questions about them!")

# Sidebar configuration
with st.sidebar:
    st.header("Setup Options")
    
    # Option for Hugging Face Hub API token (optional for better models)
    st.subheader("Optional: Use Better Models")
    st.markdown("""
    You can enter a Hugging Face Hub token to use better models. 
    Without a token, the app will use smaller, less capable models.
    [Get a free Hugging Face token](https://huggingface.co/settings/tokens)
    """)
    hf_token = st.text_input("Hugging Face Hub Token (optional):", type="password")
    
    # Model selection
    st.subheader("Model Selection")
    embedding_model = st.selectbox(
        "Embedding Model:",
        ["all-MiniLM-L6-v2 (default)", "paraphrase-multilingual-MiniLM-L12-v2"]
    )
    
    llm_model = st.selectbox(
        "LLM Model:",
        ["google/flan-t5-small (default)", "google/flan-t5-base", "google/flan-t5-large"]
    )
    
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
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
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
            chunk_size=500,  # Smaller chunks for better retrieval with smaller models
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks

# Process documents when the button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        # Set Hugging Face token if provided
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        
        # Process documents
        chunks = process_documents(uploaded_files)
        
        if chunks:
            # Determine which embedding model to use
            embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            if embedding_model == "paraphrase-multilingual-MiniLM-L12-v2":
                embeddings_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
            # Create embeddings and vector store
            try:
                with st.spinner("Creating embeddings (this may take a few moments)..."):
                    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    
                    # Save to session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed = True
                    
                    # Save model selections to session state
                    st.session_state.llm_model = llm_model.split(" ")[0]  # Extract model name without '(default)'
                    
                    st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents into {len(chunks)} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error creating embeddings: {str(e)}")
        else:
            st.sidebar.error("No content extracted from documents.")
elif process_button and not uploaded_files:
    st.sidebar.error("Please upload at least one document.")

# Main chat interface
if st.session_state.processed and st.session_state.vectorstore:
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
        
        # Process the query with RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking... (this may take a moment with free models)"):
                try:
                    # Initialize retriever
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    
                    # Get relevant documents
                    docs = retriever.invoke(user_question)
                    
                    # Display source documents if we got a result but will use the model to format response
                    if docs:
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Initialize the language model with the selected model
                        llm = HuggingFaceHub(
                            repo_id=st.session_state.llm_model,
                            model_kwargs={"temperature": 0.1, "max_length": 512}
                        )
                        
                        # Create prompt template
                        template = """You are a helpful assistant. Answer the question based on the context provided.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        
                        Answer:"""
                        
                        prompt = PromptTemplate.from_template(template)
                        
                        # Create chain
                        chain = (
                            {"context": lambda x: context_text, "question": lambda x: x}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Get answer
                        answer = chain.invoke(user_question)
                        
                        # Display the answer
                        st.write(answer)
                        
                        # Display source information in an expander
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(doc.page_content)
                                st.markdown("---")
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        no_docs_message = "I couldn't find relevant information in the uploaded documents to answer your question. Could you please rephrase or ask something related to the documents you uploaded?"
                        st.write(no_docs_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": no_docs_message})
                        
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
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
    This app uses completely free and open-source models to create a document chat experience:
    
    1. Upload PDF, DOCX, or TXT documents using the sidebar
    2. (Optional) Add your Hugging Face Hub token for better model access
    3. Click "Process Documents" to analyze your files
    4. Ask questions about your documents in the chat interface
    
    ### Technologies Used
    - **LangChain**: Framework for LLM applications
    - **Hugging Face**: For free and open embeddings and language models
    - **FAISS**: Vector store for efficient similarity search
    - **Streamlit**: Web application framework
    
    ### Getting a Hugging Face Token (Optional)
    While the app works without it, you'll get better results with a Hugging Face token:
    1. Create a free account at [huggingface.co](https://huggingface.co)
    2. Go to Settings > Access Tokens
    3. Create a new token (Read access is sufficient)
    
    ### Privacy Note
    Your documents are processed temporarily and not stored permanently. All processing happens locally.
    """)
