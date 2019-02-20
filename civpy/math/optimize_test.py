import pytest
import numpy as np
from .optimize import *


def test_fsolve():
    def func1(x):
        return x**2 + 1

    def func2(x):
        return x**2 - 1

    with pytest.raises(ValueError):
        fsolve(func1, [0])

    a = fsolve(func2, [-10, 10])
    b = np.array([-1, 1])
    assert pytest.approx(a) == b
