import numpy as np
from rpca_anomaly.solver import shrink

def test_shrink_basic():
    x = np.array([3.0, -3.0, 0.5])
    np.testing.assert_allclose(shrink(x, 1.0), [2.0, -2.0, 0.0])