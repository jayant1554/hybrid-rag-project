from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from rag.prompts import get_rag_prompt
from llm_factory import get_llm

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('file_name','?')} | Page: {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

def build_rag_chain(retriever, llm_provider: str = "ollama"):
    llm    = get_llm(llm_provider)
    prompt = get_rag_prompt()
    chain  = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def build_chain_with_sources(retriever, llm_provider: str = "ollama"):
    """Returns both answer and source documents."""
    llm    = get_llm(llm_provider)
    prompt = get_rag_prompt()
    rag    = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )
    chain_with_sources = RunnableParallel(
        answer=rag,
        sources=retriever,
    )
    return chain_with_sources
