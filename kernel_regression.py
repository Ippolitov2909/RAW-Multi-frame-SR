import numpy as np
import skimage
import time
from kernel_regression_utils import get_gauss_deriv, get_gauss_fixed_size
import scipy


from scipy.signal import correlate
from skimage.color import rgb2gray


def regress_covariances_on_eigenvals(
    dom_value,
    ratio,
    regress_params,
):
    """
    Calculates covariances of merge kernel on two orthogonal directions
    that are given by structure tensor eigenvectors based on structure tensor eigenvaluesч

    :param dom_value: dominant eigenvalue of structure tensor
    :param ratio: ratio of 1 and 2 eigenvalues of ST
    :param regress_params: parameters for regressing
    :return: k1, k2 - covariances of merge kernel on two orthogonal directions that are given by ST's eigenvectors
    """

    k_denoise = regress_params["k_denoise"]
    k_detail = regress_params["k_detail"]
    k_stretch = regress_params["k_stretch"]
    k_shrink = regress_params["k_shrink"]
    D_tr = regress_params["D_tr"]
    D_th = regress_params["D_th"]
    k_2_min = regress_params["k_2_min"]
    k_1_max = regress_params["k_1_max"]

    A = 1 + np.sqrt(ratio)
    A = np.sqrt(ratio)
    A = ratio

    D = min(max(1 + D_th - np.sqrt(dom_value) / D_tr, 0), 1)

    k1 = k_detail * k_stretch * A
    k2 = k_detail / (A * k_shrink)

    k1 = (1 - D) * k1 + D * k_detail * k_denoise
    k2 = (1 - D) * k2 + D * k_detail * k_denoise

    return min(k1, k_1_max), max(k2, k_2_min)


def covariance_regression(
    structure_tensor,
    regress_params,
):
    """

    :param structure_tensor: np.ndarray, ST in one point
    :param regress_params: parameters for covariance matrix regressing
    :return: omega (covariance matrix), k1, k2 (covariances on orthogonal directions given by ST's eigencvectors
    """

    vals, vecs = np.linalg.eig(structure_tensor)
    vecs[1, :] = -vecs[
        1:,
    ]
    if vals[1] > vals[0]:
        vecs = vecs[:, ::-1]
        vals = vals[::-1]

    k1, k2 = regress_covariances_on_eigenvals(
        dom_value=vals[0], ratio=vals[0] / vals[1], regress_params=regress_params
    )

    omega = vecs @ np.diag([k1, k2]) @ vecs.T

    return omega, k1, k2


def get_structure_tensor(img_gray, sigma_deriv=0.75, sigma_for_str_tensor=1.0):
    """
    Calculates structure tensor of grayscale image in each point
    """
    gauss_x, gauss_y = get_gauss_deriv(sigma=sigma_deriv)
    gaussian = get_gauss_fixed_size(sigma=sigma_for_str_tensor)

    grad_x = scipy.signal.convolve2d(
        skimage.color.rgb2gray(img_gray), gauss_x, mode="same"
    )
    grad_y = scipy.signal.convolve2d(
        skimage.color.rgb2gray(img_gray), gauss_y, mode="same"
    )

    grad_x_grad_x = scipy.signal.convolve2d(grad_x * grad_x, gaussian, mode="same")
    grad_x_grad_y = scipy.signal.convolve2d(grad_x * grad_y, gaussian, mode="same")
    grad_y_grad_y = scipy.signal.convolve2d(grad_y * grad_y, gaussian, mode="same")

    structure_tensor = np.array(
        [
            [
                [
                    [grad_x_grad_x[i][j], grad_x_grad_y[i][j]],
                    [grad_x_grad_y[i][j], grad_y_grad_y[i][j]],
                ]
                for j in range(img_gray.shape[1])
            ]
            for i in range(img_gray.shape[0])
        ]
    )
    return structure_tensor


def get_structure_tensors(img, sigma_deriv=0.75, sigma_for_str_tensor=1.0):
    """
    Calculates structure tensor of rgb image in each point for each color channel

    return: np.array of structure tensors, shape (img.shape[0], img.shape[1], 3, 2, 2)
    """
    img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    channels = [img_r, img_g, img_b]
    structure_tensors = [
        get_structure_tensor(channel, sigma_deriv, sigma_for_str_tensor)
        for channel in channels
    ]

    return structure_tensors


def get_map_of_covariance_matrices(
    structure_tensor,
    regress_params,
):
    """
    Calculates kernel covariances based on structure tensor
    """
    return [
        [
            covariance_regression(structure_tensor[i][j], regress_params)[0]
            for j in range(structure_tensor.shape[1])
        ]
        for i in range(structure_tensor.shape[0])
    ]


def count_kernel_val(omega_inv, y, x):
    d = np.array([[y, x]])
    return np.exp(-0.5 * d @ omega_inv @ d.T).item()


def get_point_weights(cov_matrix, flow_x, flow_y, size=3):
    omega_inv = np.linalg.inv(cov_matrix)
    weights = np.array(
        [
            [
                count_kernel_val(
                    omega_inv,
                    i - size // 2 + flow_y - np.round(flow_y),
                    j - size // 2 + flow_x - np.round(flow_x),
                )
                for j in range(size)
            ]
            for i in range(size)
        ]
    )
    return weights


def get_map_of_weights(covariance_matrices, flow, size=3):
    return np.array(
        [
            [
                get_point_weights(
                    covariance_matrices[i][j],
                    flow_x=flow[i][j][0],
                    flow_y=flow[i][j][1],
                    size=size,
                )
                for j in range(flow.shape[1])
            ]
            for i in range(flow.shape[0])
        ]
    )


def process_channel(
    channel,
    flow,
    regress_params,
    size=3,
    sigma_deriv=0.75,
    sigma_for_str_tensor=1.0,
    silent=True,
):
    structure_tensor = get_structure_tensor(
        channel, sigma_deriv=sigma_deriv, sigma_for_str_tensor=sigma_for_str_tensor
    )
    if not silent:
        print("Got structure_tensor")
    cov_matrices = get_map_of_covariance_matrices(structure_tensor, regress_params)
    if not silent:
        print("Got cov matrices")
    map_of_weights = get_map_of_weights(cov_matrices, flow, size=size)
    if not silent:
        print("Got weights")
    return map_of_weights


def process_burst(
    imgs,
    flows,
    size=3,
    sigma_deriv=0.75,
    sigma_for_str_tensor=1.0,
    silent=True,
    regress_params=None,
):
    start_time = time.time()
    maps_of_weights = []
    base = imgs[0]

    if regress_params is None:
        regress_params = {
            "k_denoise": 20,
            "k_detail": 0.15,
            "k_stretch": 1,
            "k_shrink": 2,
            "D_tr": np.sqrt(0.02),
            "D_th": 0.05,
            "k_2_min": 0.1,
            "k_1_max": 4,
        }

    structure_tensor = get_structure_tensor(
        base, sigma_deriv=sigma_deriv, sigma_for_str_tensor=sigma_for_str_tensor
    )
    if not silent:
        print("Got structure_tensor")
    cov_matrices = get_map_of_covariance_matrices(structure_tensor, regress_params)
    if not silent:
        print("Got cov matrixes")

    for i, flow in enumerate(flows):
        one_frame_start_time = time.time()
        map_of_weights = get_map_of_weights(cov_matrices, flow, size=size)
        if not silent:
            print(
                f"Got weights for {i} frame in {time.time() - one_frame_start_time} seconds"
            )
        maps_of_weights.append(map_of_weights)
    if not silent:
        print(f"Got {time.time() - start_time} seconds for processing")
    return maps_of_weights
