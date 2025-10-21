import numpy as np
import torch
from torch.nn.functional import conv2d


def convolve(matrix, kernel, stride:tuple = (1, 1), padding: tuple = (0, 0)):
    # Get heights and widths of matrix, kernel, stride and padding
    matrix_h, matrix_w = matrix.shape
    kernel_h, kernel_w = kernel.shape
    stride_y, stride_x = stride
    pad_h, pad_w = padding

    # Add padding to the matrix
    padded_matrix = np.pad(matrix, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # Calculate size of output
    output_h = (matrix_h + 2*pad_h - kernel_h)//stride_y + 1
    output_w = (matrix_w + 2*pad_w - kernel_w)//stride_x + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Calculate indexes of current region in padded matrix
            i0, j0 = i*stride_y, j*stride_x
            i1, j1 = i0 + kernel_h, j0 + kernel_w

            # Calculate and save output elment
            region = padded_matrix[i0:i1, j0:j1]
            output[i, j] = (region * kernel).sum()

    return output


def min_max_normalize(image):
    min_v = image.min(axis=(0, 1)).reshape(1, 1, 3)
    max_v = image.max(axis=(0, 1)).reshape(1, 1, 3)

    return (image - min_v) / (max_v - min_v)


def apply_filters(image, filters: dict, to_numpy: bool = True) -> dict:
    # Split the image into 3 channels
    channels = [image[:, i, :, :].unsqueeze(0) for i in range(3)]

    results = {}
    for name, filter in filters.items():
        # Apply filter to each channel of the image
        filtered_channels = [conv2d(c, filter, padding=1) for c in channels]

        # Concatenate all channels, remove the batch dimension and change the shape
        image = torch.cat(filtered_channels, dim=1).squeeze(0).permute(1, 2, 0)

        # Convert to NumPy array and set datatype to uint8
        if to_numpy:
            image = (min_max_normalize(image.numpy()) * 255).astype("uint8")

        results[name] = image

    return results
