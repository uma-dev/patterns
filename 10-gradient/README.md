# Gradient Descent

This project implements **gradient descent** for a **convex function of two variables**.

## Function

\[
f(x,y) = (x-3)^2 + 2(y+1)^2
\]

The global minimum is at **(3, −1)**.

---

## Method

Gradient descent updates:
\[
x \leftarrow x - \alpha \frac{\partial f}{\partial x}, \quad
y \leftarrow y - \alpha \frac{\partial f}{\partial y}
\]

with:

- \(\frac{\partial f}{\partial x} = 2(x-3)\)
- \(\frac{\partial f}{\partial y} = 4(y+1)\)

The algorithm stops when:

- the maximum number of epochs is reached, or
- the gradient norm is smaller than a tolerance.

---

## How to run

```bash
uv sync
PYTHONPATH=src uv run python scripts/run.py
```

## Results
<img width="857" height="750" alt="Screenshot 2026-01-09 at 7 53 16 a m" src="https://github.com/user-attachments/assets/76a2a4d7-c3ef-496c-a938-1c1fe532d8e4" />
