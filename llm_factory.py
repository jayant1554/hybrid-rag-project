import os
from config import (
    OLLAMA_MODEL,
    GROQ_API_KEY,
    NVIDIA_API_KEY
)


def get_llm(provider: str = "ollama"):

  
    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.1
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.1-8b-instant",  
            api_key=GROQ_API_KEY,
            temperature=0.1
        )
    elif provider == "mistral":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="mistral-medium-latest",   # ✅ best free/cheap model
            api_key="Q8MwodtEudJ2uJIEyCIFFoRFt8vAIod9",
            base_url="https://api.mistral.ai/v1",
            temperature=0.1,
    )
    else:
        raise ValueError(f"Unknown provider: {provider}")