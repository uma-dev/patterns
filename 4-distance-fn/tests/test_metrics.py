from distances.metrics import cosine_distance, chebyshev_distance, hamming_distance


def test_chebyshev():
    assert chebyshev_distance(
        [1, 2, 3], [4, 0, 6]) == 3.0  # max(|-3|,|2|,|-3|)=3


def test_hamming():
    assert hamming_distance([1, 0, 1, 1], [1, 1, 1, 0]) == 2


def test_cosine_basic():
    # identical -> cosine similarity 1 -> distance 0
    assert cosine_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_cosine_orthogonal():
    # [1,0] and [0,1] are orthogonal -> cos=0 -> dist=1
    assert cosine_distance([1, 0], [0, 1]) == 1.0


def test_cosine_zero_vectors():
    assert cosine_distance([0, 0], [0, 0]) == 0.0
    assert cosine_distance([0, 0], [1, 0]) == 1.0
