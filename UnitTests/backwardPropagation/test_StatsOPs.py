from QuackGrad.TensorClass import Tensor
import numpy as np


class Test_Max_Backprop:
    def test_scalar_max_backprop(self):
        a = Tensor(3.0)
        m = a.max()
        m.backwardPropagation()  
        assert np.allclose(a.grad, 1.0)

    def test_1d_max_backprop_1(self):
        a = Tensor([1.0, 5.0, 2.0])
        m = a.max() 
        m.backwardPropagation()
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert np.allclose(a.grad, expected)

    def test_1d_max_backprop_2(self):
        a = Tensor([2.0, 5.0, 5.0, 1.0])
        m = a.max()  
        m.backwardPropagation()  
        expected = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        assert np.allclose(a.grad, expected)

    def test_2d_max_1(self):
        a = Tensor([
            [1.0, 7.0, 2.0],
            [5.0, 3.0, 9.0],
        ])
        m = a.max(axis=0) 
        m.backwardPropagation()  
        expected = np.array([
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 1.0],  
        ], dtype=np.float32)
        assert np.allclose(a.grad, expected)

    def test_2d_max_2(self):
        a = Tensor([
            [3.0, 3.0, 1.0],  
            [0.0, 2.0, 2.0],   
        ])
        m = a.max(axis=1, keepdims=True)
        m.backwardPropagation()  
        expected = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ], dtype=np.float32)
        assert np.allclose(a.grad, expected)