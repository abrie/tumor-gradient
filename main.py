import fitter
import loader
import plotter

dataset = loader.load_dataset('patient_1')
data = dataset["data"]
mask = dataset["mask"]
threshold = 95.0

scaled = fitter.to_hounsfield_units(data)
masked = list(map(lambda i: fitter.apply_mask(i, mask, threshold), scaled))
grads = fitter.compute_gradients(masked)
masked_grads = list(map(lambda i: fitter.apply_mask(i, mask, threshold), grads))

slicehere = 30
scaled_slices = list(map(lambda image: image[slicehere,:,:], scaled))
masked_slices = list(map(lambda image: image[slicehere,:,:], masked))
gradient_slices = list(map(lambda image: image[slicehere,:,:], masked_grads))

plotter.plot(scaled_slices, masked_slices, gradient_slices)
plotter.show()
