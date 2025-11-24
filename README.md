# 🏙️ Dubai Estate AI Chatbot (RAG-Based)

A fully customizable **RAG (Retrieval-Augmented Generation)** chatbot built using Flask, LangChain, FAISS, and your preferred LLM (DeepSeek, OpenAI, etc.).  
This chatbot is designed for **Dubai Real Estate** queries and can be extended with your own company data.

---

## 🚀 Features

- 🔍 **RAG Pipeline** – Uses your own data (`realestate.txt`) for context-aware responses  
- 🤖 **Custom AI Personality** – Branded as *DubaiEstateBot*  
- 🧠 **LLM Compatible** – Works with DeepSeek, OpenAI, Llama, Groq, or any API  
- 💬 **Modern Chat UI** – Clean ChatGPT-like design using HTML + CSS  
- ⚡ **Fast Retrieval** powered by FAISS vector database  
- 🗂️ **Easy to Customize** – Add documents, change bot name, switch models

---

## 📁 Project Structure

rag-chatbot/
│
├── data/
│ └── realestate.txt # Your custom real-estate knowledge base
│
├── templates/
│ └── index.html # Frontend UI (ChatGPT-style)
│
├── app.py # Flask backend + LLM integration
├── rag.py # RAG logic: embeddings, vector store, retriever
├── requirements.txt # Python dependencies
└── vector_store/ # Auto-generated FAISS store (after first run)


---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ketanchauhan001/rag-chatbot.git
cd rag-chatbot
