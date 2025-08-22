from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Log:
    def test_scalar_log(self):
        a = Tensor(1)
        c = a.log()
        assert np.isclose(c.data, 0)

    def test_1d_log(self):
        a = Tensor([1, np.e, np.e**2])
        c = a.log()
        expected = np.array([0, 1, 2])
        assert np.allclose(c.data, expected)

    def test_2d_log(self):
        a = Tensor([[1, np.e], [np.e**2, np.e**3]])
        c = a.log()
        expected = np.array([[0, 1], [2, 3]])
        assert np.allclose(c.data, expected)


class Test_Mean:
    def test_scalar_mean(self):
        a = Tensor(5)
        c = a.mean()
        assert c.data == 5

    def test_1d_mean(self):
        a = Tensor([1, 3, 5, 7])
        c = a.mean()
        expected = 4
        assert np.isclose(c.data, expected)

    def test_2d_mean_axis0(self):
        a = Tensor([[1, 2], [3, 4]])
        c = a.mean(axis=0)
        expected = np.array([2, 3])
        assert np.allclose(c.data, expected)

    def test_2d_mean_axis1_keepdims(self):
        a = Tensor([[1, 2], [3, 4]])
        c = a.mean(axis=1, keepdims=True)
        expected = np.array([[1.5], [3.5]])
        assert np.allclose(c.data, expected)


class Test_Reshape:
    def test_reshape_1d_to_2d(self):
        a = Tensor(np.arange(6))
        c = a.reshape(2, 3)
        expected = np.array([[0,1,2],[3,4,5]])
        assert np.all(c.data == expected)

    def test_reshape_2d_to_1d(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        c = a.reshape(6)
        expected = np.array([1,2,3,4,5,6])
        assert np.all(c.data == expected)
