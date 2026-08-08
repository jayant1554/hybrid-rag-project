from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="bge-m3")

vec = emb.embed_query("What is hybrid rag?")

print(len(vec))