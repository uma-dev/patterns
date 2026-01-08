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

<img width="945" height="410" alt="Screenshot 2026-01-08 at 2 07 47 p m" src="https://github.com/user-attachments/assets/77f8c05b-04cb-451b-b0c6-99a64055b5d3" />

Run:

```bash
uv run python scripts/run_experiments.py

```
