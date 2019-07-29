import os
import nibabel
maskPath = os.path.join("images/mask.nii")
maskImage = nibabel.load(maskPath)
maskData = maskImage.get_fdata()
size = maskImage.shape
print(size)
