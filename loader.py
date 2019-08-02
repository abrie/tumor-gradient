import os
import re
import sys
import fnmatch
import nibabel

def find_images(path):
    result = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file,"[CBCT_]*.nii"):
            result.append(file)

    return result

def sort_images(filenames):
    regex = re.compile(r"CBCT_(\d+)\.nii")
    result = filenames.copy()
    result.sort(key=lambda filename: int(regex.match(filename).group(1)))
    return result

def load_image(path, imagename):
    imagepath = os.path.join(path, imagename)
    image = nibabel.load(imagepath)
    return image.get_fdata()

def load_images(path):
    if not os.path.exists(path):
        print(f"Cannot find folder '{os.path.abspath(path)}'\n",
                "Does the folder exist and contain patient data?"
               )
        sys.exit()

    imagenames = sort_images(find_images(path))
    return list(map(lambda name: load_image(path, name), imagenames))

def load_mask(path):
    maskPath = os.path.join(path, "mask.nii")
    maskImage = nibabel.load(maskPath)
    return maskImage.get_fdata()
