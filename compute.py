import numpy as np

def to_hounsfield_units(images):
    intercept = -1024
    return list(map(lambda image: image.copy()+intercept, images))

def compute_regressions(images, degree):
    num_images = len(images)
    A = np.stack(images)
    A2 = A.reshape(num_images, -1)
    X = np.arange(num_images)
    regressions = np.polyfit(X, A2, degree)
    newShape = images[0].shape + (degree+1,)
    return regressions.reshape(newShape)

def compute_gradients(images):
    num_images = len(images)
    A = np.stack(images)
    A2 = A.reshape(num_images, -1)
    [dx,dy] = np.gradient(A2)
    return dx.reshape(A.shape)

def apply_mask(image, mask, threshold):
    masked = np.ma.MaskedArray(image, mask < threshold)
    return masked
