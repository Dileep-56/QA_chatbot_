# QA_chatbot_
# Conversational RAG with PDF Upload and Chat History

#### Streamlit App Link - https://appchatbot-zdeuklkg6ocetot8rju4g6.streamlit.app/

This project is a **Conversational Retrieval-Augmented Generation (RAG)** application built using **Streamlit** and **LangChain**.  
It allows users to upload a PDF document and ask questions about its content, while maintaining **chat history–aware conversations**.

---

## 🚀 Features

- Upload a **PDF document**
- Ask **questions about the PDF**
- Context-aware conversations using **chat history**
- Uses **Groq LLM (LLaMA 3.1 8B Instant)**
- Vector search using **Chroma**
- Embeddings from **HuggingFace**
- Clean and interactive **Streamlit UI**

---

## 🧠 Architecture (High Level)

1. Upload PDF  
2. Split text into chunks  
3. Generate embeddings  
4. Store embeddings in Chroma  
5. Retrieve relevant chunks  
6. Generate answer using LLM  
7. Maintain session-based chat history  

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- LangChain (classic, core, community)
- Groq API
- HuggingFace Embeddings
- Chroma Vector Store
- dotenv

---

## 📁 Project Structure
```
QA_chatbot/
├── app.py                       # Streamlit app (entry point)
├── rag.py                       # RAG pipeline logic
├── llm.py                       # Groq + embeddings
├── vectorstore.py               # Chroma logic
├── prompts.py                   # Prompt templates
├── session.py                   # Chat history
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd <project-folder>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Create a .env file:

HF_TOKEN=your_huggingface_token


## Groq API Key is to be entered directly in the Streamlit UI.

▶️ Run the App
streamlit run code_assistant_app.py

```
