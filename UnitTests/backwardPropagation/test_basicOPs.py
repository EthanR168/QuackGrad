from QuackGrad.TensorClass import Tensor
import numpy as np

def test_add_backprop():
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x + y
    z.backwardPropagation()
    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, 1.0)

def test_sub_backprop():
    x = Tensor(5.0)
    y = Tensor(3.0)
    z = x - y
    z.backwardPropagation()
    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, -1.0)

def test_mul_backprop():
    x = Tensor(2.0)
    y = Tensor(4.0)
    z = x * y
    z.backwardPropagation()
    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)

def test_div_backprop():
    x = Tensor(8.0)
    y = Tensor(2.0)
    z = x / y
    z.backwardPropagation()
    assert np.allclose(x.grad, 1 / y.data)
    assert np.allclose(y.grad, -x.data / (y.data ** 2))

def test_exp_backprop():
    x = Tensor(np.array([0.0, 1.0, 2.0]), requiresGrad=True)
    y = x.exp()          
    y.backwardPropagation(np.ones_like(y.data))  
    expected_grad = np.exp(x.data)  
    assert np.allclose(x.grad, expected_grad)