from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Max:
    def test_ScalarMax(self):
        a = Tensor(3)
        c = a.max()
        assert c.data == 3

    def test_1DMax(self):
        a = Tensor([1, 5, 2, 4])
        c = a.max()  
        assert c.data == 5

    def test_2DMax_1(self):
        a = Tensor([
            [1, 2],
            [7, 4],
        ])
        c = a.max()  
        assert c.data == 7

    def test_2DMax_2(self):
        a = Tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        c0 = a.max(axis=0)
        expected0 = np.array([4, 5, 6])
        assert np.all(c0.data == expected0)

        c1 = a.max(axis=1, keepdims=True)
        expected1 = np.array([[3], [6]])
        assert np.all(c1.data == expected1)

    def test_3DMax(self):
        a = Tensor([
            [
                [1, 7],
                [3, 2],
            ],
            [
                [5, 4],
                [6, 8],
            ],
        ])  
        c_axis0 = a.max(axis=0)
        expected_axis0 = np.array([
            [5, 7],
            [6, 8],
        ])
        assert np.all(c_axis0.data == expected_axis0)

        c_axis2 = a.max(axis=2)
        expected_axis2 = np.array([
            [7, 3],
            [5, 8],
        ])
        assert np.all(c_axis2.data == expected_axis2)