import matplotlib.pyplot as plt

import fitter
import loader

def show_slices(slices, masks, gradients):
    fig, axes = plt.subplots(3, len(slices), sharey=True, sharex=True)
    for i, s in enumerate(slices):
        axes[0][i].imshow(s, cmap="gray", origin="lower")

    for i, s in enumerate(masks):
        axes[1][i].imshow(s, cmap="gray", origin="lower")

    for i, s in enumerate(gradients):
        axes[2][i].imshow(s, cmap=plt.cm.jet, origin="lower")

images = loader.loadImages()
mask = loader.loadMask()

raw = list(map(lambda i: i["data"], images))
masked = list(map(lambda i: fitter.apply_mask(i, mask, 95.0), raw))
grads = fitter.compute_gradients(masked)
masked_grads = list(map(lambda i: fitter.apply_mask(i, mask, 95.0), grads))

slicehere = 30
ss = list(map(lambda image: image[slicehere,:,:], raw))
ms = list(map(lambda image: image[slicehere,:,:], masked))
gs = list(map(lambda image: image[slicehere,:,:], masked_grads))
show_slices(ss, ms, gs)

plt.show()
