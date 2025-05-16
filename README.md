# RAG Document Chat Application

A Streamlit application that allows users to upload documents (PDF, DOCX, TXT) and chat with them using Retrieval Augmented Generation (RAG).

## Features

- Document upload and processing (PDF, DOCX, TXT)
- Free and open-source models from Hugging Face
- No API keys required (optional Hugging Face token for better performance)
- Interactive chat interface
- Source document references for transparency

## Demo

You can see a live demo of this application on Streamlit Cloud: [Your Streamlit App URL]

## Installation

```bash
git clone https://github.com/yourusername/document-chat-app.git
cd document-chat-app
pip install -r requirements.txt
streamlit run app.py
```

## How to Use

1. Upload your documents (PDF, DOCX, TXT)
2. (Optional) Add a free Hugging Face token for better model performance
3. Click "Process Documents"
4. Ask questions about your documents in the chat interface

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for LLM applications
- **Hugging Face**: For free and open embeddings and language models
- **FAISS**: Vector store for efficient similarity search

## Getting a Free Hugging Face Token (Optional)

While the app works without it, you can get better results with a free Hugging Face token:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings > Access Tokens
3. Create a new token (Read access is sufficient)

## Deployment

This app can be deployed on Streamlit Cloud for free:

1. Push this code to a GitHub repository
2. Sign up for a free account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy by pointing to your GitHub repository

## License

MIT License
