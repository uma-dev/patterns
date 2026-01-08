# Gaussian Naïve Bayes

This report summarizes the experimental results of a **Gaussian Naïve Bayes (GNB)** classifier implemented **from scratch** and evaluated using **Stratified 10-Fold Cross-Validation** and **Leave-One-Out (LOO)** validation.

---

## Model

- Gaussian Naïve Bayes with conditional independence assumption
- Log-likelihood formulation for numerical stability
- Variance smoothing: `ε = 1e-9`
- No feature scaling or normalization

---

## Datasets

- **Iris** (3 classes, 150 samples)
- **Wine** (3 classes, 178 samples)
- **Breast Cancer Wisconsin** (2 classes, 569 samples)

`sklearn` was used **only** for dataset loading.

---

## Validation

- **Stratified 10-Fold Cross-Validation** (explicit implementation, seed = 42)
- **Leave-One-Out (LOO)**

---

## Metrics

Computed using **one-vs-rest (OVR)** with **macro-averaging**:

- Sensitivity (TPR)
- Specificity (TNR)
- Balanced Accuracy = (Sensitivity + Specificity) / 2

---

## Results

### Stratified 10-Fold Cross-Validation

| Dataset       |  Balanced Accuracy |        Sensitivity |        Specificity |
| ------------- | -----------------: | -----------------: | -----------------: |
| Iris          | _(program output)_ | _(program output)_ | _(program output)_ |
| Wine          | _(program output)_ | _(program output)_ | _(program output)_ |
| Breast Cancer | _(program output)_ | _(program output)_ | _(program output)_ |

### Leave-One-Out (LOO)

| Dataset       |  Balanced Accuracy |        Sensitivity |        Specificity |
| ------------- | -----------------: | -----------------: | -----------------: |
| Iris          | _(program output)_ | _(program output)_ | _(program output)_ |
| Wine          | _(program output)_ | _(program output)_ | _(program output)_ |
| Breast Cancer | _(program output)_ | _(program output)_ | _(program output)_ |

---

## Key Observations

- GNB performs well on low-dimensional, well-separated datasets.
- LOO results are slightly more stable but computationally expensive.
- Balanced Accuracy provides a fair comparison across binary and multiclass problems.

---

## Reproducibility

- Python + `uv`
- Libraries: `numpy`, `scikit-learn` (datasets only)
- All models and validations implemented from scratch
