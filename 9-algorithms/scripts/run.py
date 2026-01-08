from __future__ import annotations

from compare.datasets import DATASETS
from compare.experiment import evaluate_dataset
from compare.validation import CVConfig


def main() -> int:
    cfg = CVConfig(n_splits=10, seed=42)

    print("# Classification Algorithms (Financial)\n")
    print("Configuration:")
    print(f"- Validation: Stratified {cfg.n_splits}-Fold CV (seed={cfg.seed})")
    print("- Metrics: Sensitivity, Specificity, Balanced Accuracy, F1-score\n")

    for ds in DATASETS:
        print(f"\n## {ds.name}\n")
        try:
            res = evaluate_dataset(ds, cfg)
        except Exception as e:
            print(f"Failed to evaluate '{ds.openml_name}': {e}")
            continue

        print("| Model | Sensitivity | Specificity | Balanced Acc | F1-score |")
        print("|---|---:|---:|---:|---:|")
        for model_name, m in res.items():
            print(
                f"| {model_name} | {m.sensitivity:.4f} | {m.specificity:.4f} | {m.balanced_acc:.4f} | {m.f1:.4f} |"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
