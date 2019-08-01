import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np

import fitter
import loader

def show_slices(slices, masks, gradients):
    data_min = np.amin(np.stack(slices))
    data_max = np.amax(np.stack(slices))
    data_bounds = [data_min, data_max]
    data_norm = colors.Normalize(vmin=data_min, vmax=data_max)
    data_colormap = "gray"

    mask_colormap = data_colormap
    mask_norm = data_norm

    gradient_min = np.amin(np.stack(gradients))
    gradient_max = np.amax(np.stack(gradients))
    gradient_bounds = [gradient_min, 0, gradient_max]
    gradient_colormap = colors.ListedColormap(['green', 'red'])
    gradient_norm = colors.BoundaryNorm(gradient_bounds, gradient_colormap.N)

    fig, axes = plt.subplots(3, len(slices), sharey=True, sharex=True)
    for i, s in enumerate(slices):
        axes[0][i].imshow(s, cmap=data_colormap, origin="lower")

    for i, s in enumerate(masks):
        axes[1][i].imshow(s, cmap=mask_colormap, origin="lower")

    for i, s in enumerate(gradients):
        img = axes[2][i].imshow(s, cmap=gradient_colormap, norm=gradient_norm, origin="lower")

    fig.colorbar(cm.ScalarMappable(cmap=data_colormap, norm=gradient_norm), ax=axes[0], use_gridspec=True)
    fig.colorbar(cm.ScalarMappable(cmap=mask_colormap, norm=mask_norm), ax=axes[1], use_gridspec=True)
    fig.colorbar(cm.ScalarMappable(cmap=gradient_colormap, norm=gradient_norm), ax=axes[2], use_gridspec=True)

images = loader.loadImages()
mask = loader.loadMask()
masked = list(map(lambda i: fitter.apply_mask(i, mask, 95.0), images))
grads = fitter.compute_gradients(masked)
masked_grads = list(map(lambda i: fitter.apply_mask(i, mask, 95.0), grads))

slicehere = 30
ss = list(map(lambda image: image[slicehere,:,:], images))
ms = list(map(lambda image: image[slicehere,:,:], masked))
gs = list(map(lambda image: image[slicehere,:,:], masked_grads))
show_slices(ss, ms, gs)

plt.show()
