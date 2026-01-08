# Decision Tree (Gini) — Results Summary

Binary decision tree implemented from scratch (CART-style splits `feature <= threshold`) using **Gini impurity**.

## Setup

- Python + `uv`
- Libraries: `numpy`, `scikit-learn` (datasets only)

## Experiment

- Validation: **stratified hold-out 70/30**, seed = 42
- Metric: **accuracy**
- Tree: `max_depth = 4`, criterion = Gini

## Results

Run:

```bash
uv run python scripts/run_experiments.py

```
