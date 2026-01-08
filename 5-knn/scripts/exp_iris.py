from __future__ import annotations
import argparse
from knn import KNNClassifier, StandardScaler, Pipeline
from knn.data import load_iris
from knn.cv import KFold, LeaveOneOut, cross_validate
from knn.metrics import sensitivity, specificity, balanced_accuracy, per_class_sensitivity_specificity
from knn.reporting import format_folds_table, format_summary_table, format_per_class_table

def make_pipeline(k: int):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", KNNClassifier(k=k))])

def main():
    ap = argparse.ArgumentParser(description="Iris experiments: K-Fold & LOO with KNN")
    ap.add_argument("--k", type=int, default=3, help="neighbors")
    ap.add_argument("--n-splits", type=int, default=5, help="KFold splits")
    ap.add_argument("--loo", action="store_true", help="Run Leave-One-Out in addition to KFold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, class_names = load_iris()

    metrics = {
        "macro_sensitivity": lambda yt, yp: sensitivity(yt, yp, average="macro"),
        "macro_specificity": lambda yt, yp: specificity(yt, yp, average="macro"),
        "balanced_accuracy": balanced_accuracy,  # macro recall
    }

    # K-Fold
    print(f"\n=== K-Fold (n_splits={args.n_splits}, k={args.k}) ===")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    pipe_factory = lambda: make_pipeline(args.k)
    res_kf = cross_validate(pipe_factory, X, y, kf, metrics)
    print(format_folds_table(res_kf["folds"]))
    print()
    print(format_summary_table(res_kf["summary"]))

    # Per-class (on full-fit for interpretability)
    pipe = make_pipeline(args.k).fit(X, y)
    yhat = pipe.predict(X)
    rec_c, spec_c, _ = per_class_sensitivity_specificity(y, yhat)
    print("\nPer-class sensitivity (train-fit, interpretive only):")
    print(format_per_class_table(rec_c, class_names, "class"))
    print("\nPer-class specificity (train-fit, interpretive only):")
    print(format_per_class_table(spec_c, class_names, "class"))

    # Optional LOO
    if args.loo:
        print(f"\n=== Leave-One-Out (k={args.k}) ===")
        loo = LeaveOneOut()
        res_loo = cross_validate(pipe_factory, X, y, loo, metrics)
        print(format_summary_table(res_loo["summary"]))

if __name__ == "__main__":
    main()
