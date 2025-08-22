from QuackGrad.TensorClass import Tensor
import numpy as np

def test_softmax_backprop():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x.softmax()
    grad_out = np.array([1.0, 0.0, 0.0])
    y.backwardPropagation(grad_out)
    exps = np.exp(x.data - np.max(x.data))
    softmax_out = exps / np.sum(exps)
    dot = np.sum(grad_out * softmax_out)
    expected_grad = softmax_out * (grad_out - dot)
    assert np.allclose(x.grad, expected_grad)

def test_relu_backprop():
    x = Tensor(np.array([-1.0, 0.0, 2.0]))
    y = x.ReLU()
    y.backwardPropagation()
    expected_grad = np.array([0.0, 0.0, 1.0])
    assert np.allclose(x.grad, expected_grad)