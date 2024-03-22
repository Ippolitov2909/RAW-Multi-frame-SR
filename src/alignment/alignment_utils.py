import numpy as np
from scipy.signal import fftconvolve
from skimage.measure import block_reduce


def calc_d2(template, extended):
    """
    Calculates cross-correlation between template and extended image for all shifts with FFT
    Needed for block-based optical flow
    """
    try:
        cross_correlation = fftconvolve(
            extended, template[::-1, ::-1], mode="same", axes=(0, 1)
        ).sum(axis=2)
        box = np.ones(template.shape)
        extended_squared = fftconvolve(
            extended ** 2, box, mode="same", axes=(0, 1)
        ).sum(axis=2)
    except Exception:
        print(extended, template)
        cross_correlation = 0
        extended_squared = np.zeros(template.shape[0:2])
    d2_distance = -2 * cross_correlation + extended_squared + (template ** 2).sum()
    return d2_distance

def get_filter_gauss(sigma, size):
    gaussian = np.array(
        [
            [
                np.exp(
                    -((i - size // 2) ** 2 + (j - size // 2) ** 2) / (2 * sigma ** 2)
                )
                for j in range(0, size)
            ]
            for i in range(0, size)
        ]
    )
    gaussian /= gaussian.sum()

    return np.array([gaussian, gaussian, gaussian]).T


def downsample(image):
    gaussian = get_filter_gauss(sigma=1.2, size=5)
    blurred = fftconvolve(image, gaussian, mode="same", axes=(0, 1))
    downsampled = block_reduce(image, (2, 2, 1), func=np.mean)
    return downsampled


def extract(
    base_frame, image, flow, row_block_index, column_block_index, block_size=16
):
    # flow = flow_x, flow_y
    # o___x
    # |
    # |
    # y
    # picture bounds must pe processed accurately

    flow = flow.astype(int)
    up = max(block_size * row_block_index, 0)
    down = min(up + block_size, base_frame.shape[0])
    left = max(block_size * column_block_index, 0)
    right = min(left + block_size, base_frame.shape[1])
    tpl = base_frame[up:down, left:right]

    up_new = max(up + flow[1], 0)
    left_new = max(left + flow[0], 0)
    down_new = min(down + flow[1], base_frame.shape[0])
    right_new = min(right + flow[0], base_frame.shape[1])
    crop = image[up_new:down_new, left_new:right_new]

    if crop.shape != tpl.shape:
        return None, None
    return tpl, crop
