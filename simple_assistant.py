import os
from operator import itemgetter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Updated Import Path
from langchain_classic.memory import ConversationBufferMemory 

# 1. SETUP THE LIBRARIAN
print("📚 Loading ONLY your project files...")
loader = DirectoryLoader('./', glob="*.py", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load() 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"✅ Successfully split {len(docs)} files into {len(chunks)} smart chunks.")

# 2. CREATE THE SEARCHABLE LIBRARY
print("🔍 Indexing documents into ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
retriever = vector_db.as_retriever()

# 3. SETUP THE BRAIN & MEMORY
llm = OllamaLLM(model="llama3", timeout=120)
# 'memory_key' MUST match the name in your template
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key="answer", input_key="question")

template = """
You are a Senior Python Developer and AI Architect. 
Use the context and chat history to answer the question professionally.

Context:
{context}

Chat History:
{chat_history}

Question: 
{question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 4. THE CHAIN
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. RUN IT
print("\n🤖 Assistant is ready!")

# Q1
q1 = "What is the main database I am using?"
print(f"\nUser: {q1}")
res1 = chain.invoke({"question": q1})
print(f"AI: {res1}")
memory.save_context({"question": q1}, {"answer": res1})

# Q2 - THE MEMORY TEST
q2 = "Why is it a good choice for this project?"
print(f"\nUser: {q2}")
res2 = chain.invoke({"question": q2})
print(f"AI: {res2}")