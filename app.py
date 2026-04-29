import streamlit as st
import os
from operator import itemgetter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory

# --- 1. PAGE CONFIG & UI ---
st.set_page_config(page_title="CodeTrace AI", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ CodeTrace AI")
st.caption("Your local project expert. I've read your code and I'm ready to help.")
st.markdown("---")

# --- 2. THE LIBRARIAN (RAG Setup) ---
@st.cache_resource
def init_rag():
    loader = DirectoryLoader(
        './', 
        glob="*.py", 
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs={'encoding': 'utf-8'} 
    )
    docs = loader.load()
    
    if not docs:
        st.error("I couldn't find any Python files in this folder!")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever()

retriever = init_rag()

# --- 3. THE BRAIN & MEMORY ---
llm = OllamaLLM(model="llama3", timeout=120)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, 
        memory_key="chat_history", 
        output_key="answer", 
        input_key="question"
    )

my_memory = st.session_state.memory

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. THE "HUMAN" PROMPT ---
# We changed the instructions here to be more friendly and less robotic.
template = """
You are CodeTrace AI, a friendly and expert developer partner. 
Your goal is to help the user understand their project by looking at the provided code.

RULES:
1. Be conversational and helpful, like a teammate sitting next to the user.
2. If the answer is in the code below, explain it clearly in plain English.
3. Don't just list technical functions; explain *what* they do for the user.
4. If you don't know the answer based on the code, just say so—don't guess.

Project Context:
{context}

Chat History:
{chat_history}

User's Question: 
{question}

CodeTrace Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": lambda x: my_memory.load_memory_variables({})["chat_history"]
    }
    | prompt | llm | StrOutputParser()
)

# --- 5. THE CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything about your files..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking your files..."):
            response = chain.invoke({"question": user_input})
            st.markdown(response)
            
            my_memory.save_context({"question": user_input}, {"answer": response})
            st.session_state.messages.append({"role": "assistant", "content": response})