# 4- Distance Functions

3 distance functions that accept two vectors (arrays) and return a distance:

- Cosine distance
- Chebyshev distance
- Hamming distance

## Setup (uv)

```bash
uv init --name distance-functions
uv add numpy
uv add --dev pytest
```

## Code

```bash
uv run python scripts/distcli.py cosine --a "1,2,3" --b "4,5,6"
uv run python scripts/distcli.py chebyshev --a "1,2,3" --b "4,5,6"
uv run python scripts/distcli.py hamming --a "1,0,1,1" --b "1,1,1,0"
```

## Pytest

```bash
uv run pytest
```
