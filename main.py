import compute
import loader
import plotter

dataset = loader.load_dataset('patient_1')
data = dataset["data"]
mask = dataset["mask"]
threshold = 95.0

scaled = compute.to_hounsfield_units(data)
masked = list(map(lambda i: compute.apply_mask(i, mask, threshold), scaled))
gradients = compute.compute_gradients(masked)
masked_gradients = list(
    map(lambda i: compute.apply_mask(i, mask, threshold), gradients))


def slicer(arr): return arr[30, :, :]


scaled_slices = list(map(slicer, scaled))
masked_slices = list(map(slicer, masked))
gradient_slices = list(map(slicer, masked_gradients))

plotter.plot(scaled_slices, masked_slices, gradient_slices)
plotter.show()
