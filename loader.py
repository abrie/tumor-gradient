import os
import nibabel


def loadImage(path):
    image = nibabel.load(path)
    data = image.get_fdata()
    shape = image.shape
    return {"image":image,"data":data,"shape":shape}

def loadImages():
    imageFiles = list(map(lambda i: f'images/CBCT_{i}.nii', range(1,8)))
    return list(map(lambda path: loadImage(path), imageFiles))

def loadMask():
    maskPath = os.path.join("images/mask.nii")
    maskImage = nibabel.load(maskPath)
    return maskImage.get_fdata()
