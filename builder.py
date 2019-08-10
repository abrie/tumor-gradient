import os

import numpy as np
import nibabel

def write_nifti(path, named, data):
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, os.path.join(path,named))

def generate_nifti_file(path, named, shape, fill_value):
    data = np.full(shape, fill_value, dtype=np.int16)
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, os.path.join(path,named))
