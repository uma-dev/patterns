from __future__ import annotations

import argparse

from distances.metrics import cosine_distance, chebyshev_distance, hamming_distance
from distances.utils import parse_csv_vector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute distances between two vectors (comma-separated)."
    )
    sub = parser.add_subparsers(dest="metric", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--a", required=True, help='Vector a, e.g. "1,2,3"')
        p.add_argument("--b", required=True, help='Vector b, e.g. "4,5,6"')

    p_cos = sub.add_parser("cosine", help="Cosine distance")
    add_common_args(p_cos)

    p_cheb = sub.add_parser(
        "chebyshev", help="Chebyshev distance (L-infinity)")
    add_common_args(p_cheb)

    p_ham = sub.add_parser(
        "hamming", help="Hamming distance (# of mismatches)")
    add_common_args(p_ham)

    args = parser.parse_args()

    a = parse_csv_vector(args.a)
    b = parse_csv_vector(args.b)

    if args.metric == "cosine":
        d = cosine_distance(a, b)
    elif args.metric == "chebyshev":
        d = chebyshev_distance(a, b)
    elif args.metric == "hamming":
        d = hamming_distance(a, b)
    else:
        raise RuntimeError("Unknown metric")

    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
