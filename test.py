import unittest
import numpy as np
import fitter
import loader
import nibabel
import os

def generate_Nifiti_testfile(path, named, fill_value):
    data = np.full((3,3,3), fill_value, dtype=np.int16)
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, os.path.join(path,named))

def generate_testfiles(path, count):
    if not os.path.exists(path):
        os.mkdir(path)
    for x in range(1,count+1):
        filename = f"CBCT_{x}.nii"
        fill_value = x
        generate_Nifiti_testfile(path, filename, fill_value)

class TestAllTheThings(unittest.TestCase):
    def setUp(self):
        self.vals = [0.0,1.0,5.0,7.0]
        self.samples = list(map(lambda p: np.zeros((3, 3, 3)), self.vals))
        for (sample, val) in zip(self.samples, self.vals):
            sample[0][0][0] = val

        mask = np.zeros((3, 3, 3))
        mask[0][0][0] = 50.0
        self.mask = mask
        self.mask_threshold = mask[0][0][0]-1 # Set threshold to data-1, leaving it unmasked.

        self.testfiles_dir = "testdata"
        self.testfiles_count = 10
        generate_testfiles(self.testfiles_dir, self.testfiles_count)

    def test_imagefinder(self):
        pathlist = loader.find_images(self.testfiles_dir)
        self.assertEqual(len(pathlist), self.testfiles_count)

    def test_imageloader(self):
        data = loader.load_images(self.testfiles_dir)
        self.assertEqual(len(data), self.testfiles_count)
        self.assertEqual(data[0][0][0][0], 1)
        self.assertEqual(data[self.testfiles_count-1][0][0][0], self.testfiles_count)

    def test_listsorter(self):
        unsorted = ["CBCT_2.nii","CBCT_15.nii","CBCT_3.nii"]
        expected = ["CBCT_2.nii","CBCT_3.nii","CBCT_15.nii"]
        result = loader.sort_images(unsorted)
        self.assertEqual(result, expected)

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
