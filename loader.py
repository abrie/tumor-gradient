import os
import nibabel


def loadImage(path):
    image = nibabel.load(path)
    return image.get_fdata()

def loadImages():
    imagepaths = list(map(lambda i: f'images/CBCT_{i}.nii', range(1,8)))
    return list(map(lambda path: loadImage(path), imagepaths))

def loadMask():
    maskPath = os.path.join("images/mask.nii")
    maskImage = nibabel.load(maskPath)
    return maskImage.get_fdata()
