import os
import math
import numpy as np
import builder


def generate_testfiles(path, shape, count):
    if not os.path.exists(path):
        os.mkdir(path)

    for x in range(1, count+1):
        named = f"CBCT_{x}.nii"
        data = np.random.choice([math.sin((2*math.pi/3)*x-1)*100], shape)
        builder.write_nifti(path, named, data)

    builder.generate_nifti_file(path, "mask.nii", shape, 95)


generate_testfiles("input", (64, 64, 64), 7)
