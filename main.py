import fitter
import loader
import plotter

data = loader.load_images('patient_1')
mask = loader.load_mask('patient_1')
threshold = 95.0

data = fitter.to_hounsfield_units(data)
masked = list(map(lambda i: fitter.apply_mask(i, mask, threshold), data))
grads = fitter.compute_gradients(masked)
masked_grads = list(map(lambda i: fitter.apply_mask(i, mask, threshold), grads))

slicehere = 30
data_slices = list(map(lambda image: image[slicehere,:,:], data))
masked_slices = list(map(lambda image: image[slicehere,:,:], masked))
gradient_slices = list(map(lambda image: image[slicehere,:,:], masked_grads))

plotter.plot(data_slices, masked_slices, gradient_slices)
plotter.show()
