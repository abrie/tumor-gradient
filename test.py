import unittest
import numpy as np
import fitter

class TestStacks(unittest.TestCase):
    def setUp(self):
        self.vals = [0,1,5,7]
        self.samples = list(map(lambda p: np.zeros((3, 3, 3)), self.vals))
        for (sample, val) in zip(self.samples, self.vals):
            sample[0][0][0] = val

    def test_fitter(self):
        degree = 1
        computed = fitter.compute_regressions(self.samples, degree)
        expected = np.polyfit(range(len(self.samples)), self.vals, degree)
        self.assertAlmostEqual(computed[0][0][0][0], expected[0], places=3)

    def test_gradient(self):
        gradients = fitter.compute_gradients(self.samples)
        self.assertEqual(gradients[0][0][0][0], 1.0)
        self.assertEqual(gradients[1][0][0][0], 2.5)
        self.assertEqual(gradients[2][0][0][0], 3.0)
        self.assertEqual(gradients[3][0][0][0], 2.0)


if __name__ == '__main__':
    unittest.main()
