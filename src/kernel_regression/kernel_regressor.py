import time
import numpy as np

from src.kernel_regression.kernel_constructor import KernelConstructor
from scipy.signal import convolve2d
from skimage.color import rgb2gray

from src.kernel_regression.kernel_regression_utils import get_gauss_deriv, get_gauss_fixed_size



class KernelRegressor:
    def __init__(
        self,
        k_denoise: float = 20,
        k_detail: float = 0.15,
        k_stretch: float = 1,
        k_shrink: float = 2,
        D_tr: float = np.sqrt(0.02),
        D_th: float = 0.05,
        k_2_min: float = 0.1,
        k_1_max: float = 4,
        sigma_deriv=0.75,
        sigma_for_str_tensor=1.0
    ):
        self.k_denoise = k_denoise
        self.k_detail = k_detail
        self.k_stretch = k_stretch
        self.k_shrink = k_shrink
        self.D_tr = D_tr
        self.D_th = D_th
        self.k_2_min = k_2_min
        self.k_1_max = k_1_max
        self.sigma_deriv = sigma_deriv
        self.sigma_for_str_tensor = sigma_for_str_tensor

    def _regress_covariances_on_eigenvals(
        self,
        dom_value,
        ratio,
    ):
        """
        Calculates covariances of merge kernel on two orthogonal directions
        that are given by structure tensor eigenvectors based on structure tensor eigenvaluesч

        :param dom_value: dominant eigenvalue of structure tensor
        :param ratio: ratio of 1 and 2 eigenvalues of ST
        :return: k1, k2 - covariances of merge kernel on two orthogonal directions that are given by ST's eigenvectors
        """

        A = 1 + np.sqrt(ratio)
        A = np.sqrt(ratio)
        A = ratio

        D = min(max(1 + self.D_th - np.sqrt(dom_value) / self.D_tr, 0), 1)

        k1 = self.k_detail * self.k_stretch * A
        k2 = self.k_detail / (A * self.k_shrink)

        k1 = (1 - D) * k1 + D * self.k_detail * self.k_denoise
        k2 = (1 - D) * k2 + D * self.k_detail * self.k_denoise

        return min(k1, self.k_1_max), max(k2, self.k_2_min)
    
    
    def covariance_regression(
        self,
        structure_tensor,
    ):
        """
        Calculates covariance matrix of a given ST in one point

        :param structure_tensor: np.ndarray, ST in one point
        :return: omega (covariance matrix), k1, k2 (covariances on orthogonal directions given by ST's eigencvectors
        """
        # eigendecomposition
        vals, vecs = np.linalg.eig(structure_tensor)
        # sorting eigenvalues and their vectors
        vecs[1, :] = -vecs[
            1:,
        ]
        if vals[1] > vals[0]:
            vecs = vecs[:, ::-1]
            vals = vals[::-1]

        # regression of covariances
        k1, k2 = self._regress_covariances_on_eigenvals(
            dom_value=vals[0], ratio=vals[0] / vals[1]
        )

        # constructing covariance matrix
        omega = vecs @ np.diag([k1, k2]) @ vecs.T

        return omega, k1, k2

    def get_structure_tensor(self, img):
        """
        Calculates structure tensor of rgb image in each point after converting it to grayscale
        """
        gauss_x, gauss_y = get_gauss_deriv(sigma=self.sigma_deriv)
        gaussian = get_gauss_fixed_size(sigma=self.sigma_for_str_tensor)

        img_gray = rgb2gray(img) if len(img.shape) == 3 and img.shape[2] == 3 else img
        grad_x = convolve2d(img_gray, gauss_x, mode="same")
        grad_y = convolve2d(img_gray, gauss_y, mode="same")

        grad_x_grad_x = convolve2d(grad_x * grad_x, gaussian, mode="same")
        grad_x_grad_y = convolve2d(grad_x * grad_y, gaussian, mode="same")
        grad_y_grad_y = convolve2d(grad_y * grad_y, gaussian, mode="same")

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
    
    def get_structure_tensors_from_rgb(self, img):
        """
        Calculates structure tensor of rgb image in each point for each color channel

        return: np.array of structure tensors, shape (img.shape[0], img.shape[1], 3, 2, 2)
        """
        img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        channels = [img_r, img_g, img_b]
        structure_tensors = [
            self.get_structure_tensor(channel, self.sigma_deriv, self.sigma_for_str_tensor)
            for channel in channels
        ]

        return structure_tensors
    
    def get_map_of_covariance_matrices(
        self,
        structure_tensor,
    ):
        """
        Calculates kernel covariances based on structure tensor
        """
        return [
            [
                self.covariance_regression(structure_tensor[i][j])[0]
                for j in range(structure_tensor.shape[1])
            ]
            for i in range(structure_tensor.shape[0])
        ]
    
    def process_channel(
        self,
        channel,
        flow,
        size=3,
        silent=True,
    ):
        structure_tensor = self.get_structure_tensor(channel)
        if not silent:
            print("Got structure_tensor")
        cov_matrices = self.get_map_of_covariance_matrices(structure_tensor)
        if not silent:
            print("Got cov matrices")
        map_of_weights = KernelConstructor.get_map_of_weights(cov_matrices, flow, size=size)
        if not silent:
            print("Got weights")
        return map_of_weights
    
    def process_burst(
        self,
        imgs,
        flows,
        size=3,
        silent=True,
    ):
        start_time = time.time()
        maps_of_weights = []
        base = imgs[0]

        structure_tensor = self.get_structure_tensor(base)
        if not silent:
            print("Got structure_tensor")
        cov_matrices = self.get_map_of_covariance_matrices(structure_tensor)
        if not silent:
            print("Got cov matrixes")

        for i, flow in enumerate(flows):
            one_frame_start_time = time.time()
            map_of_weights = KernelConstructor.get_map_of_weights(cov_matrices, flow, size=size)
            if not silent:
                print(
                    f"Got weights for {i} frame in {time.time() - one_frame_start_time} seconds"
                )
            maps_of_weights.append(map_of_weights)
        if not silent:
            print(f"Got {time.time() - start_time} seconds for processing")
        return maps_of_weights

