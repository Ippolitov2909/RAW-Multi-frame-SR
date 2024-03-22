import numpy as np

from tqdm import tqdm
from skimage.exposure import adjust_gamma

class MergeProcessor:
    def __init__(self):
        pass

    def merge(self, imgs, flows, weights, masks=None, verbose=False):
        """
        imgs - list of RAW images
        flows - list of optical flows
        weights - list of kernels for each image
        """

        rounded_flows = flows.copy().round()
        img_shape = imgs[0].shape
        print(img_shape)
        kernel_size = weights.shape[-1]

        if masks is None:
            masks = [np.ones((img_shape[0], img_shape[1])) for i in range(len(imgs))]

        merged_frame = np.zeros(imgs[0].shape)
        for row in tqdm(range(img_shape[0])) if verbose else range(img_shape[0]):
            for col in range(img_shape[1]):
                for channel in range(3):
                    weights_sum = 0
                    value = 0

                    for img_index, frame in enumerate(imgs):
                        kernel = weights[img_index][row][col].copy()
                        
                        point_neighbours = np.zeros(kernel.shape)
                        flow_x, flow_y = rounded_flows[img_index][row][col]
                        flow_x = int(flow_x)
                        flow_y = int(flow_y)

                        up = left = 0
                        down = right = kernel_size
                        if row + flow_y < kernel_size // 2:
                            up = kernel_size // 2 - row - flow_y
                        elif row + flow_y + kernel_size // 2 + 1 > img_shape[0]:
                            down = img_shape[0] - row - kernel_size // 2 - 1 - flow_y

                        if col + flow_x < kernel_size // 2:
                            left = kernel_size // 2 - col - flow_x
                        elif col + flow_x + kernel_size // 2 + 1 > img_shape[1]:
                            right = img_shape[1] - col - kernel_size // 2 - 1 - flow_x

                        point_neighbours[up:down, left:right] = frame[min(max(row + flow_y - kernel_size // 2, 0),
                                                                        img_shape[0]): max(
                            min(row + flow_y + kernel_size // 2 + 1, img_shape[0]), 0),
                                                                min(max(col + flow_x - kernel_size // 2, 0),
                                                                    img_shape[1]): max(
                                                                    min(col + flow_x + kernel_size // 2 + 1, img_shape[1]),
                                                                    0),
                                                                channel]
                        kernel *= (point_neighbours != 0)

                        weights_sum += kernel.sum() * masks[img_index][row][col]
                        value += (kernel * point_neighbours).sum() * masks[img_index][row][col]

                    if weights_sum > 0:
                        value /= weights_sum

                    merged_frame[row][col][channel] = value

        return merged_frame

