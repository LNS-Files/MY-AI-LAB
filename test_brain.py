from langchain_ollama import OllamaLLM

# Use the updated class name
llm = OllamaLLM(model="llama3")

response = llm.invoke("What is a variable in programming?")
print(response)