import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np

def show():
    plt.show()

def plot(data, masks, gradients):
    fig, axes = plt.subplots(3, len(data), sharey=True, sharex=True)

    # plot the data slices
    data_stack = np.stack(data)
    data_min = np.amin(data_stack)
    data_max = np.amax(data_stack)
    data_bounds = [data_min, data_max]
    data_norm = colors.Normalize(vmin=data_min, vmax=data_max)
    data_colormap = "gray"

    fig.colorbar(cm.ScalarMappable(cmap=data_colormap, norm=data_norm), ax=axes[0], use_gridspec=True)

    for i, s in enumerate(data):
        axes[0][i].imshow(s, cmap=data_colormap, norm=data_norm, origin="lower")


    # plot the mask slices
    mask_colormap = data_colormap
    mask_norm = data_norm

    fig.colorbar(cm.ScalarMappable(cmap=mask_colormap, norm=mask_norm), ax=axes[1], use_gridspec=True)

    for i, s in enumerate(masks):
        axes[1][i].imshow(s, cmap=mask_colormap, norm=mask_norm, origin="lower")


    # plot the gradient slices
    gradient_stack = np.stack(gradients)
    gradient_min = np.amin(gradient_stack)
    gradient_max = np.amax(gradient_stack)
    gradient_bounds = [gradient_min, 0, gradient_max]
    #gradient_colormap = colors.ListedColormap(['green', 'red'])
    gradient_colormap = colors.LinearSegmentedColormap.from_list('gradient_colormap', [ (0, 1, 0), (1, 0, 0) ], N=25)
    gradient_norm = colors.Normalize(vmin=gradient_min, vmax=gradient_max)

    fig.colorbar(cm.ScalarMappable(cmap=gradient_colormap, norm=gradient_norm), ax=axes[2], use_gridspec=True)

    for i, s in enumerate(gradients):
        img = axes[2][i].imshow(s, cmap=gradient_colormap, norm=gradient_norm, origin="lower")
