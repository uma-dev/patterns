## Report: Effect of **k** in K-NN

### 1) Why does K-NN classify **100% correctly when k = 1**, but not when **k ≠ 1**?

This depends on **what is being evaluated**:

- **Training**:  
  **k = 1**, every training sample’s nearest neighbor is **itself** (distance = 0), so it will always be assigned its own label. Therefore, training accuracy becomes **100%**, thats memorization.  
  **k > 1**, the prediction for a training sample is influenced by **other nearby samples**, so decisions are smoothed. If there are neighbors from other classes nearby (overlap, noise, boundary points), majority voting can flip the label, so training accuracy may drop below 100%.

- **Validation/test**:  
  **k = 1**, is typically **high variance** and may overfit, thats performing worse on unseen points.

---

### 2) What happens to the decision boundary when **k is very small**?

- **Low bias, high variance**
- Tends to **overfit**
- The classifier becomes **highly sensitive** to individual samples.
- The decision boundary is:
  - **very irregular**
  - small “islands” around outliers
  - affected by noise and mislabeled points

---

### 3) What happens to the decision boundary when **k is very large**?

- Tends to **underfit** if k is too large (ignores meaningful local patterns).
- **Higher bias, lower variance**
- Predictions depend on a big neighborhood, possibly spanning many regions.
- The decision boundary becomes:
  - **smoother**
  - more dominated by the global class distribution
  - less responsive

---

### 4) What happens if **k is greater than the number of patterns in two classes (Iris)** or greater than the number of patterns in a class (binary classification)?

#### 4.1 Multiclass

- `N` = total training samples
- `n_a` = number of samples in class `a`
- `n_c` = number of samples in class `c`

If **k > n_a + n_b** (sum of patterns in two classes), then the neighbor set must include points from the **third class** as well
So:

- Cannot select k neighbors only from two classes if their total is smaller than k.
- The voting is increasingly influenced by **all classes**, not just the locally relevant ones.
- The classifier’s output starts drifting toward the **global majority** or toward the class with higher representation.
- Solution: `k ≤ N_train`.

#### 4.2 Binary classification

- The class with more samples will be favored more often.
- Distance weighting strongly favors closer points.
- As k grows large, the classifier behaves like a **prior-based classifier**:
  - predicts the **majority class** most of the time.

- Solution: `1 ≤ k ≤ N_train`
- Hint: odd k in binary problems to reduce ties.
- Scaling/normalization; so no feature dominate distance.

---

### Bias–Variance Tradeoff

- **Small k** → complex boundary → overfit (high variance)
- **Large k** → smooth boundary → underfit (high bias)
- **k close to N** → predicts majority class almost everywhere
- **k=1 gives 100% only on training set** because each point is its own nearest neighbor
