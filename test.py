import unittest
import numpy as np
import fitter
import loader

class TestAllTheThings(unittest.TestCase):
    def setUp(self):
        self.vals = [0.0,1.0,5.0,7.0]
        self.samples = list(map(lambda p: np.zeros((3, 3, 3)), self.vals))
        for (sample, val) in zip(self.samples, self.vals):
            sample[0][0][0] = val

        mask = np.zeros((3, 3, 3))
        mask[0][0][0] = 50.0
        self.mask = mask
        self.mask_threshold = mask[0][0][0]-1 # Set threshold below the data value, to leave it unmasked.

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

    def test_masking(self):
        sample = self.samples[3]
        masked = fitter.apply_mask(sample, self.mask, self.mask_threshold)
        self.assertEqual(np.ma.is_masked(masked),True)
        self.assertEqual(masked[0][0][0],sample[0][0][0])
        self.assertEqual(masked.mask[0][0][0],False)
        self.assertEqual(masked.mask[1][0][0],True)

if __name__ == '__main__':
    unittest.main()
