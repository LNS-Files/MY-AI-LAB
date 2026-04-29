from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Connect to the Librarian (Memory)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 2. Connect to the Brain (Llama 3)
llm = OllamaLLM(model="llama3")

# 3. Create the Prompt Template
# This tells the Brain exactly how to use the context we give it
template = """Answer the user's question based only on the following context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Build the modern Retrieval Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 5. Ask the Assistant!
response = retrieval_chain.invoke({"input": "How can this project help a developer?"})

print(f"\nAI Assistant: {response['answer']}")