from __future__ import annotations
import argparse
import json
from typing import List
import numpy as np

from . import KNNClassifier, StandardScaler, Pipeline
from .data import load_iris
from .cv import KFold, LeaveOneOut, cross_validate
from .metrics import sensitivity, specificity, balanced_accuracy
from .reporting import format_folds_table, format_summary_table

# ---- helpers ----

def _make_pipeline(k: int, metric_name: str) -> Pipeline:
    from .distances import l2, l1, cosine
    metrics = {"l2": l2, "l1": l1, "cosine": cosine}
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric '{metric_name}'. Use one of {list(metrics)}")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNNClassifier(k=k, metric=metrics[metric_name])),
    ])

def _metrics_dict():
    return {
        "macro_sensitivity": lambda yt, yp: sensitivity(yt, yp, average="macro"),
        "macro_specificity": lambda yt, yp: specificity(yt, yp, average="macro"),
        "balanced_accuracy": balanced_accuracy,
    }

# ---- subcommands ----

def cmd_cv(args: argparse.Namespace) -> int:
    if args.dataset == "iris":
        X, y, _ = load_iris()
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    def pipe_factory():
        return _make_pipeline(args.k, args.metric)

    metrics = _metrics_dict()

    if args.fold == "kfold":
        splitter = KFold(n_splits=args.n_splits, shuffle=not args.no_shuffle, random_state=args.seed)
    elif args.fold == "loo":
        splitter = LeaveOneOut()
    else:
        raise SystemExit("fold must be one of {'kfold','loo'}")

    res = cross_validate(pipe_factory, X, y, splitter, metrics)
    print(format_folds_table(res["folds"]))
    print()
    print(format_summary_table(res["summary"]))
    return 0

def cmd_predict(args: argparse.Namespace) -> int:
    # tiny convenience: load iris, fit on all, predict a single vector
    X, y, _ = load_iris()
    pipe = _make_pipeline(args.k, args.metric).fit(X, y)

    # parse features
    if args.features_json:
        feats = np.array(json.loads(args.features_json), dtype=float)
    elif args.features_csv:
        feats = np.fromstring(args.features_csv, sep=",", dtype=float)
    else:
        raise SystemExit("Provide --features-csv 'f1,f2,...' or --features-json '[...]'")

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    preds = pipe.predict(feats)
    try:
        probas = pipe.predict_proba(feats).tolist()
    except AttributeError:
        probas = None

    # print compact JSON to make it script-friendly
    out = {"pred": preds.tolist()}
    if probas is not None:
        out["proba"] = probas
    print(json.dumps(out))
    return 0

# ---- main ----

def main():
    ap = argparse.ArgumentParser(prog="knn", description="KNN-from-scratch CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # cv
    ap_cv = sub.add_parser("cv", help="Cross-validate on a dataset (iris)")
    ap_cv.add_argument("--dataset", default="iris", choices=["iris"])
    ap_cv.add_argument("--fold", default="kfold", choices=["kfold","loo"])
    ap_cv.add_argument("--n-splits", type=int, default=5, help="KFold splits (ignored for LOO)")
    ap_cv.add_argument("--no-shuffle", action="store_true", help="Disable shuffling for KFold")
    ap_cv.add_argument("--seed", type=int, default=42)
    ap_cv.add_argument("--k", type=int, default=3, help="Number of neighbors")
    ap_cv.add_argument("--metric", default="l2", choices=["l2","l1","cosine"])
    ap_cv.set_defaults(func=cmd_cv)

    # predict
    ap_pred = sub.add_parser("predict", help="Fit on iris and predict a feature vector")
    ap_pred.add_argument("--k", type=int, default=3)
    ap_pred.add_argument("--metric", default="l2", choices=["l2","l1","cosine"])
    g = ap_pred.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-csv", help="Comma-separated features 'f1,f2,...'")
    g.add_argument("--features-json", help="JSON array of features, e.g. '[5.1,3.5,1.4,0.2]'")
    ap_pred.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    raise SystemExit(args.func(args))

if __name__ == "__main__":
    main()
