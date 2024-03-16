import numpy as np

from scipy.signal import fftconvolve

class RobustnessProcessor:
    def __init__(
            self, 
            blur_size:int = 5,
            size_for_min: int = 5,
            s: float = 1.75,
            d: float = np.exp(-2),
            v: float = 3.0

    ):
        self.blur_size = blur_size
        self.size_for_min = size_for_min
        self.s = s
        self.d = d
        self.v = v

    def get_variance(self, img, mean=None):
        if mean is None:
            mean = fftconvolve(img, np.ones((self.blur_size, self.blur_size, 1)) / (self.blur_size ** 2), mode='same', axes=(0, 1))
        variance = np.zeros((mean.shape[0], mean.shape[1]))
        for row in range(mean.shape[0]):
            for col in range(mean.shape[1]):
                variance[row][col] = ((img[max(row - self.blur_size // 2, 0): row + self.blur_size // 2 + 1,
                                    col - self.blur_size // 2: col + self.blur_size // 2 + 1] - mean[row][col]) ** 2).sum()
                variance[row][col] /= self.blur_size ** 2

        return variance


    def get_diff(self, img, extra_img, flow, size=3):
        diff = np.zeros((flow.shape[0], flow.shape[1]))
        flow = np.round(flow).astype(int)
        for row in range(flow.shape[0]):
            for col in range(flow.shape[1]):
                flow_x, flow_y = flow[row][col]
                i = min(max(row + flow_y, 0), flow.shape[0] - 1)
                j = min(max(col + flow_x, 0), flow.shape[1] - 1)
                diff[row][col] = (img[row][col] - extra_img[i][j]).mean()
                if row == 495 and col == 495:
                    print(flow_x, flow_y)

        return fftconvolve(diff, np.ones((size, size)) / (size ** 2), axes=(0, 1), mode='same')


    def get_robustness_mask(
            self,
            img, extra_img, flow, 
    ):
        """

        :param img: base image, np.ndarray
        :param extra_img: extra image, np.ndarray
        :param flow: optical flow between base image and extra image
        :param size: size of neighbourhood in which variance and diff are counted
        :param size_for_min: size of neighbourhood in which minimum robustness is chosen
        :param s: param for mask, float
        :param d: param for mask, float
        :param v: param for mask, float
        :return: robustness mask, np.ndarray with shape (img.shape[0], img.shape[1])
        """
        variance = self.get_variance(img)

        diff = self.get_diff(img, extra_img, flow)

        mask = self.s * np.exp(- self.v * diff ** 2 / (variance + 0.001)) - self.d

        mask[mask > 1] = 1
        mask[mask < 0] = 0

        new_mask = np.zeros(mask.shape)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                new_mask[row][col] = mask[max(row - self.size_for_min // 2, 0): row + self.size_for_min // 2 + 1,
                                    max(col - self.size_for_min // 2, 0): col + self.size_for_min // 2 + 1].min()

        return new_mask
