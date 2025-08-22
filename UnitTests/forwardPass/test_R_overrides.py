from QuackGrad.TensorClass import Tensor
import numpy as np

def test_rmul():
    a = 5
    b = Tensor([
        [1, 2],
        [3, 4],
    ])
    c = a * b
    expected = Tensor([
        [5, 10],
        [15, 20],
    ])
    assert np.allclose(c.data, expected.data)

def test_rtruediv():
    a = 5
    b = Tensor([
        [1, 2],
        [4, 5],
    ])
    c = a / b
    expected = Tensor([
        [5, 2.5],
        [1.25, 1],
    ])
    assert np.allclose(c.data, expected.data)

def test_radd():
    a = 5
    b = Tensor([
        [1, 2],
        [3, 4],
    ])
    c = a + b
    expected = Tensor([
        [6, 7],
        [8, 9],
    ])
    assert np.allclose(c.data, expected.data)

def test_rsub():
    a = 5
    b = Tensor([
        [1, 2],
        [3, 4],
    ])
    c = a - b
    expected = Tensor([
        [4, 3],
        [2, 1],
    ])
    assert np.allclose(c.data, expected.data)

def test_rfloordiv():
    a = 5
    b = Tensor([
        [1, 2],
        [3, 4],
    ])
    c = a // b
    expected = Tensor([
        [5, 2],
        [1, 1],
    ])
    assert np.allclose(c.data, expected.data)

def test_rmod():
    a = 5
    b = Tensor([
        [1, 2],
        [3, 4],
    ])
    c = a % b
    expected = Tensor([
        [0, 1],
        [2, 1],
    ])
    assert np.allclose(c.data, expected.data)
    
def test_rmatmul():
    a = 2
    b = Tensor(3)
    c = a @ b
    assert c.data == 6
