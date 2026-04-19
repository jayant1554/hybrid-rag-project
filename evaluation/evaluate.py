import json
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from rag.chain import build_rag_chain
from config import ALPHA, EMBED_MODEL


# 🔥 Local Ollama (no API limits)
ragas_llm = LangchainLLMWrapper(
    OllamaLLM(model="llama3", temperature=0)
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model=EMBED_MODEL)
)


def run_evaluation(mode: str = "hybrid_rerank", alpha: float = ALPHA):
    # 🔹 Load dataset
    with open("evaluation/sample_test.json") as f:
        test_data = json.load(f)

    # 🔹 Retriever
    retriever = get_hybrid_retriever(alpha=alpha)

    if mode == "hybrid_rerank":
        retriever = get_reranker(retriever)

    # 🔹 RAG chain
    chain = build_rag_chain(retriever)

    results = []

    for item in test_data:
        q = item["question"]

        # 🔥 Get answer
        answer = chain.invoke(q)

        # ✅ Ensure string output
        if hasattr(answer, "content"):
            answer = answer.content
        else:
            answer = str(answer)

        # 🔥 Retrieve docs
        docs = retriever.invoke(q)

        # ✅ CLEAN CONTEXTS (THIS FIXES NaN ISSUE)
        contexts = []
        for d in docs:
            if d.page_content:
                text = d.page_content.strip()
                if len(text) > 30:   # remove weak chunks
                    contexts.append(text)

        # 🔥 Fallback (critical)
        if not contexts:
            contexts = ["No relevant context found"]

        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    # 🔹 Convert to dataset
    dataset = Dataset.from_list(results)

    # 🔹 Evaluate
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print(f"\n=== RAGAS Results ({mode}, alpha={alpha}) ===")
    print(scores)

    # 🔹 Save results
    with open(f"evaluation/results_{mode}.json", "w") as f:
        json.dump(
            scores.to_pandas().to_dict(orient="records"),
            f,
            indent=2
        )

    return scores


if __name__ == "__main__":
    run_evaluation()