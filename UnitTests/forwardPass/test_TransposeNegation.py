from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Transpose:
    def test_2d_transpose(self):
        a = Tensor([[1, 2], [3, 4]])
        c = a.T
        expected = np.array([[1, 3], [2, 4]])
        assert np.all(c.data == expected)

    def test_3d_transpose(self):
        a = Tensor(np.arange(8).reshape(2,2,2))
        c = a.T
        expected = np.transpose(np.arange(8).reshape(2,2,2))
        assert np.all(c.data == expected)


class Test_Negation:
    def test_negate_scalar(self):
        a = Tensor(5)
        c = -a
        assert c.data == -5

    def test_negate_1d(self):
        a = Tensor([1, -2, 3])
        c = -a
        expected = np.array([-1, 2, -3])
        assert np.all(c.data == expected)

    def test_negate_2d(self):
        a = Tensor([[1, -2], [3, -4]])
        c = -a
        expected = np.array([[-1, 2], [-3, 4]])
        assert np.all(c.data == expected)