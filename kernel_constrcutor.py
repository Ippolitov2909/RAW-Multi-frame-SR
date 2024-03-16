import numpy as np

class KernelConstructor:
    def __init__(self, size=3):
        pass

    @staticmethod
    def count_kernel_val(omega_inv, y, x):
        """
        Calculates the value of the kernel function (pdf from normal distribution with covariance matrix omega_inv) at point (y, x)
        """
        d = np.array([[y, x]])
        return np.exp(-0.5 * d @ omega_inv @ d.T).item()

    @staticmethod
    def construct_kernel(cov_matrix, flow_x, flow_y, size=3):
        """
        Calculates the weights of the kernel function (pdf from normal distribution with covariance matrix omega_inv) 
        Each weight corresponds to the point calculated taking optical flow into account
        """
        omega_inv = np.linalg.inv(cov_matrix)
        weights = np.array(
            [
                [
                    KernelConstructor.count_kernel_val(
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
    
    @staticmethod
    def get_map_of_weights(covariance_matrices, flow, size=3):
        return np.array(
            [
                [
                    KernelConstructor.construct_kernel(
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