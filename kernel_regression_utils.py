import numpy as np


def gauss(x, y, sigma=3.0):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / np.sqrt(
        2 * np.pi * sigma ** 2
    )


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


def get_gauss_deriv(sigma=3.0):
    half_size = int(4 * sigma)
    deriv_y = np.array(
        [
            [
                gauss(i - half_size, j - half_size, sigma)
                * 2
                * (half_size - i)
                / (2 * sigma ** 2)
                for j in range(2 * half_size + 1)
            ]
            for i in range(2 * half_size + 1)
        ]
    )
    deriv_x = np.array(
        [
            [
                gauss(i - half_size, j - half_size, sigma)
                * 2
                * (half_size - j)
                / (2 * sigma ** 2)
                for j in range(2 * half_size + 1)
            ]
            for i in range(2 * half_size + 1)
        ]
    )
    return deriv_x, deriv_y


def get_gauss_fixed_size(sigma=3.0, size=4):
    gaussian = np.array(
        [
            [
                np.exp(-((y - size) ** 2 + (x - size) ** 2) / (2 * sigma ** 2))
                for x in range(2 * size + 1)
            ]
            for y in range(2 * size + 1)
        ]
    )
    gaussian /= gaussian.sum()
    gaussian.sum()
    return gaussian

