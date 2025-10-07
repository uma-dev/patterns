from __future__ import annotations
from typing import List, Dict

def _pad(s, w): return str(s).ljust(w)

def format_folds_table(folds: List[dict]) -> str:
    """
    Returns a simple text table of per-fold metrics.
    """
    if not folds:
        return "(no folds)"
    # collect metric names
    metric_names = list(folds[0]["metrics"].keys())
    headers = ["fold", "n_train", "n_test"] + metric_names
    colw = {h: max(len(h), 8) for h in headers}

    for f in folds:
        colw["fold"] = max(colw["fold"], len(str(f["idx"])))
        colw["n_train"] = max(colw["n_train"], len(str(f["n_train"])))
        colw["n_test"] = max(colw["n_test"], len(str(f["n_test"])))
        for m in metric_names:
            colw[m] = max(colw[m], len(f"{f['metrics'][m]:.4f}"))

    lines = []
    lines.append(" | ".join(_pad(h, colw[h]) for h in headers))
    lines.append("-+-".join("-" * colw[h] for h in headers))
    for f in folds:
        row = [f["idx"], f["n_train"], f["n_test"]] + [f"{f['metrics'][m]:.4f}" for m in metric_names]
        lines.append(" | ".join(_pad(v, colw[h]) for v, h in zip(row, headers)))
    return "\n".join(lines)

def format_summary_table(summary: Dict[str, Dict[str, float]]) -> str:
    headers = ["metric", "mean", "std"]
    colw = {h: max(len(h), 8) for h in headers}
    for m, agg in summary.items():
        colw["metric"] = max(colw["metric"], len(m))
        colw["mean"] = max(colw["mean"], len(f"{agg['mean']:.4f}"))
        colw["std"]  = max(colw["std"],  len(f"{agg['std']:.4f}"))

    lines = []
    lines.append(" | ".join(_pad(h, colw[h]) for h in headers))
    lines.append("-+-".join("-" * colw[h] for h in headers))
    for m, agg in summary.items():
        lines.append(" | ".join([
            _pad(m, colw["metric"]),
            _pad(f"{agg['mean']:.4f}", colw["mean"]),
            _pad(f"{agg['std']:.4f}",  colw["std"]),
        ]))
    return "\n".join(lines)

def format_per_class_table(values, labels, title: str) -> str:
    """
    values: array-like per-class numbers; labels: list of class names/ids.
    """
    headers = [title, "value"]
    colw = {h: max(len(h), 8) for h in headers}
    lines = []
    lines.append(" | ".join(_pad(h, colw[h]) for h in headers))
    lines.append("-+-".join("-" * colw[h] for h in headers))
    for lab, val in zip(labels, values):
        lines.append(" | ".join([_pad(str(lab), colw[headers[0]]), _pad(f"{val:.4f}", colw["value"])]))
    return "\n".join(lines)
