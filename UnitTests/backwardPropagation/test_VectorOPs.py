from QuackGrad.TensorClass import Tensor
import numpy as np

def test_matmul_backprop():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = Tensor(np.array([[2.0, 0.0], [1.0, 2.0]]))
    z = x @ y
    grad_out = np.ones_like(z.data)
    z.backwardPropagation(grad_out)
    assert np.allclose(x.grad, grad_out @ y.data.T)
    assert np.allclose(y.grad, x.data.T @ grad_out)

def test_sum_backprop():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x.sum()
    y.backwardPropagation()
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_mean_backprop():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x.mean()
    y.backwardPropagation()
    assert np.allclose(x.grad, np.ones_like(x.data) * (1/3))

def test_reshape_backprop():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = x.reshape(4)
    y.backwardPropagation(np.ones(4))
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_transpose_backprop():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = x.T
    grad_out = np.ones_like(y.data)
    y.backwardPropagation(grad_out)
    assert np.allclose(x.grad, grad_out.T)