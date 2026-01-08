# Classification Algorithms (Financial Domain)

This assignment compares several **classification algorithms from scikit-learn** on **financial datasets**, focusing on understanding how different models behave under the same evaluation conditions.

---

## Datasets

Financial / credit datasets loaded from **OpenML**:

- German Credit
- Australian Credit Approval
- Credit Approval
- Default of Credit Card Clients
- Bank Marketing

All problems are treated as **binary classification**.

---

## Algorithms

- Logistic Regression
- k-Nearest Neighbors (k=5)
- Decision Tree
- Gaussian Naïve Bayes

---

## Validation

- **Stratified 10-Fold Cross-Validation**
- Fixed random seed: `42`

---

## Metrics

Computed per fold and averaged:

- **Sensitivity**
- **Specificity**
- **Balanced Accuracy**
- **F1-score**

---

## How to Run

```bash
uv sync
PYTHONPATH=src uv run python scripts/run.py
```

## Results
