
import time
from evaluation.evaluate import run_evaluation

modes = [
    ("dense_only", 1.0),
    ("sparse_only", 0.0),
    ("hybrid", 0.4),
    ("hybrid_rerank", 0.4),
]

print("\n" + "="*65)
print(f"{'Mode':<20} {'Faithful':>9} {'Relevancy':>10} {'Precision':>10} {'Recall':>7}")
print("="*65)

for mode, alpha in modes:
    print(f"\nRunning: {mode}")

    scores = run_evaluation(mode=mode, alpha=alpha)
    scores = scores.to_pandas().to_dict(orient="records")[0]

    print(
        f"{mode:<20} "
        f"{scores['faithfulness']:>9.3f} "
        f"{scores['answer_relevancy']:>10.3f} "
        f"{scores['context_precision']:>10.3f} "
        f"{scores['context_recall']:>7.3f}"
    )

    time.sleep(3)
print("="*65)