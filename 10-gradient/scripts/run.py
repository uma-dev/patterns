from __future__ import annotations

import argparse

from gd import GDConfig, gradient_descent
from gd.functions import f_xy


def print_table(history, n=5) -> None:
    if not history:
        print("No history recorded.")
        return

    head = history[:n]
    tail = history[-n:] if len(history) > n else []

    def row(s):
        return f"| {s.epoch} | {s.x:.6f} | {s.y:.6f} | {s.f:.6f} | {s.grad_norm:.6e} |"

    print("| epoch | x | y | f(x,y) | ||grad|| |")
    print("|---:|---:|---:|---:|---:|")
    for s in head:
        print(row(s))

    if tail:
        print("| ... | ... | ... | ... | ... |")
        for s in tail:
            print(row(s))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Gradient Descent on a convex function of two variables")
    p.add_argument("--x0", type=float, default=0.0, help="Initial x")
    p.add_argument("--y0", type=float, default=0.0, help="Initial y")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--epochs", type=int, default=200, help="Max iterations")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Early stop if ||grad|| < tol")
    args = p.parse_args()

    cfg = GDConfig(lr=args.lr, epochs=args.epochs, tol=args.tol)
    x, y, history = gradient_descent(args.x0, args.y0, cfg)

    print("# Task 10 — Gradient Descent (2 variables)\n")
    print("Function:")
    print("- f(x,y) = (x-3)^2 + 2(y+1)^2")
    print("\nConfig:")
    print(f"- lr={cfg.lr}, epochs={cfg.epochs}, tol={cfg.tol}\n")

    print("## Final result")
    print(f"- epochs_used: {len(history)}")
    print(f"- x*: {x:.8f}")
    print(f"- y*: {y:.8f}")
    print(f"- f(x*,y*): {f_xy(x,y):.10f}\n")

    print("## Iteration preview (first/last 5)")
    print_table(history, n=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
