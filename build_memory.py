from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load and Chunk (What we did before)
loader = TextLoader("./README.md")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 2. Setup the Translator (Embeddings)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Setup the Librarian (Vector Store)
# This creates a folder called 'db' to save your memory
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Memory bank created! The Librarian has filed the chunks.")