from __future__ import annotations

from dtree.datasets import load_all_datasets
from dtree.model import DecisionTreeBinary
from dtree.validation import HoldoutConfig, evaluate_holdout_accuracy


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    cfg = HoldoutConfig(test_size=0.30, seed=42)
    max_depth = 4

    rows = []
    for ds in load_all_datasets():
        acc = evaluate_holdout_accuracy(
            ds.X,
            ds.y,
            config=cfg,
            model_factory=lambda: DecisionTreeBinary(max_depth=max_depth),
        )
        rows.append((ds.name, acc))

    # GitHub-renderable markdown
    print("# Decision Tree (Gini) — Holdout 70/30 Results\n")
    print("Configuration:")
    print(
        f"- Split: stratified hold-out (train 70% / test 30%), seed={cfg.seed}")
    print(
        f"- Model: binary decision tree (CART-style), criterion=Gini, max_depth={max_depth}\n")

    print("| Dataset | Accuracy |")
    print("|---|---:|")
    for name, acc in rows:
        print(f"| {name} | {fmt(acc)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
