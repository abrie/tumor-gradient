import os
import re
import fnmatch
import nibabel

def findImages(path):
    result = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file,"[CBCT_]*.nii"):
            result.append(file)

    return result

def sortImages(filenames):
    regex = re.compile(r"CBCT_(\d+)\.nii")
    result = filenames.copy()
    result.sort(key=lambda filename: int(regex.match(filename).group(1)))
    return result

def loadImage(path):
    image = nibabel.load(path)
    return image.get_fdata()

def loadImages(path):
    filelist = findImages(path)
    imagepaths = list(map(lambda filename: os.path.join(path, filename), filelist))
    return list(map(lambda path: loadImage(path), imagepaths))

def loadMask(path):
    maskPath = os.path.join(path, "mask.nii")
    maskImage = nibabel.load(maskPath)
    return maskImage.get_fdata()
