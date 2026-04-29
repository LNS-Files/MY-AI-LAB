# 🕵️‍♂️ CodeTrace AI: Local Project Expert

**CodeTrace AI** is a professional-grade RAG (Retrieval-Augmented Generation) assistant that transforms your local Python codebase into an interactive knowledge base. Built for privacy and speed, it allows you to "chat" with your files to understand complex logic, find bugs, or document features instantly.

---

## 🚀 Key Features
* **100% Privacy:** Powered by **Ollama**, your code never leaves your local machine.
* **Semantic Search:** Uses **ChromaDB** and **Vector Embeddings** to find context even when you don't use exact keywords.
* **Contextual Memory:** Remembers your previous questions to provide deep-dive architectural advice.
* **Professional UI:** A clean, responsive dashboard built with **Streamlit**.

---

## 🛠️ The Tech Stack
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Brain:** [Ollama](https://ollama.com/) (Llama 3)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** `nomic-embed-text`
* **Frontend:** [Streamlit](https://streamlit.io/)

---

## 🧠 How it Works
This project implements a standard **RAG Pipeline**:
1. **Ingestion:** Scans local `.py` files using `DirectoryLoader`.
2. **Chunking:** Breaks code into 1,000-character segments with `RecursiveCharacterTextSplitter`.
3. **Indexing:** Converts chunks into vectors and stores them in a local ChromaDB collection.
4. **Retrieval:** When asked a question, it retrieves the most relevant code snippets.
5. **Generation:** Passes the code snippets + chat history to Llama 3 to generate a human-like response.

---

## ⚙️ Installation & Setup

### 1. Install Ollama
Download from [ollama.com](https://ollama.com/) and run:
```bash
ollama pull llama3
ollama pull nomic-embed-text