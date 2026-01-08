from __future__ import annotations

from gnb.datasets import load_all_datasets
from gnb.model import GaussianNB
from gnb.validation import CVConfig, run_kfold_cv, run_leave_one_out


def fmt(x: float) -> str:
    # fixed formatting for report tables
    return f"{x:.4f}"


def print_markdown_table(title: str, rows: list[tuple[str, float, float, float]]) -> None:
    print(f"\n## {title}\n")
    print("| Dataset | Balanced Accuracy | Sensitivity | Specificity |")
    print("|---|---:|---:|---:|")
    for name, bal, sens, spec in rows:
        print(f"| {name} | {fmt(bal)} | {fmt(sens)} | {fmt(spec)} |")


def main() -> int:
    datasets = load_all_datasets()

    # Explicit stratified 10-fold
    cv_cfg = CVConfig(n_splits=10, seed=42)

    cv_rows = []
    loo_rows = []

    for ds in datasets:
        # fresh model each time
        cv_res = run_kfold_cv(ds.X, ds.y, config=cv_cfg,
                              model_factory=lambda: GaussianNB(eps=1e-9))
        loo_res = run_leave_one_out(
            ds.X, ds.y, model_factory=lambda: GaussianNB(eps=1e-9))

        cv_rows.append((ds.name, cv_res.balanced_accuracy,
                       cv_res.sensitivity, cv_res.specificity))
        loo_rows.append((ds.name, loo_res.balanced_accuracy,
                        loo_res.sensitivity, loo_res.specificity))

    print("# Gaussian Naive Bayes — Results\n")
    print("Configuration:")
    print("- Model: GaussianNB (from scratch), eps=1e-9")
    print(
        f"- CV: Stratified 10-fold (explicit implementation), seed={cv_cfg.seed}")
    print("- Preprocessing: none (no scaling/standardization)\n")

    print_markdown_table("Stratified 10-Fold Cross-Validation", cv_rows)
    print_markdown_table("Leave-One-Out (LOO)", loo_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
