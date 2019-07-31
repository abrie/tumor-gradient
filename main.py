import matplotlib.pyplot as plt

import fitter
import loader

def show_slices(slices, gradients):
    fig, axes = plt.subplots(2, len(slices), sharey=True, sharex=True)
    for i, s in enumerate(slices):
        axes[0][i].imshow(s, cmap="gray", origin="lower")

    for i, s in enumerate(gradients):
        axes[1][i].imshow(s, cmap=plt.cm.jet, origin="lower")

#show_slices(list(map(lambda image: image["data"][35,:,:], images)))

datalist = list(map(lambda image: image["data"], images))
gradients = fitter.compute_gradients(datalist)

slicehere = 30
ss = list(map(lambda image: image["data"][slicehere,:,:], images))
gs = list(map(lambda image: image[slicehere,:,:], gradients))
show_slices(ss, gs)

plt.show()
