from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the document
loader = TextLoader("./README.md")
docs = loader.load()

# 2. Setup the "Slicer"
# chunk_size: How big each piece is (characters)
# chunk_overlap: How much to repeat from the previous piece
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
)

# 3. Chop it up!
chunks = text_splitter.split_documents(docs)

print(f"I turned 1 document into {len(chunks)} chunks.")
print(f"Here is the first chunk: {chunks[0].page_content}")