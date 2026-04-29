from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Connect to the existing Librarian
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# 2. Ask a question
query = "What is the goal of this project?"

# 3. Search the memory!
# k=2 tells the librarian to bring us the top 2 best matches
results = vector_db.similarity_search(query, k=2)

print(f"I found {len(results)} relevant chunks.")
for i, res in enumerate(results):
    print(f"Chunk {i+1}: {res.page_content}")