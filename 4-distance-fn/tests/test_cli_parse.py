import numpy as np
from distances.utils import parse_csv_vector


def test_parse_csv_vector():
    v = parse_csv_vector("1, 2,3")
    assert np.allclose(v, np.array([1.0, 2.0, 3.0]))
