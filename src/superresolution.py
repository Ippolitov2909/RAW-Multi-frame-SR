import time
import numpy as np 
from src.alignment.alignment import OpticalFlowProcessor
# from alignment import OpticalFlowProcessor
from src.kernel_regression.kernel_regressor import KernelRegressor
from src.robustness import RobustnessProcessor
from src.merge import MergeProcessor
from src.utils.utils import upsample_weights
from src.utils.raw_utils import make_sparse_raw

class SuperResolutionProcessor:
    def __init__(
            self,
            depth: int = 3,
        k_denoise: float = 20,
        k_detail: float = 0.15,
        k_stretch: float = 1,
        k_shrink: float = 2,
        D_tr: float = np.sqrt(0.02),
        D_th: float = 0.05,
        k_2_min: float = 0.1,
        k_1_max: float = 4,
        sigma_deriv=0.75,
        sigma_for_str_tensor=1.0,
        s=1.5, d=1.5 * np.exp(-2), v=2.5
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
        self.opt_flow_processor = OpticalFlowProcessor(
            depth=depth
        )
        self.kernel_regressor = KernelRegressor(k_denoise=k_denoise, k_detail=k_detail, k_stretch=k_stretch, k_shrink=k_shrink, D_tr=D_tr, D_th=D_th, k_2_min=k_2_min, k_1_max=k_1_max, sigma_deriv=sigma_deriv, sigma_for_str_tensor=sigma_for_str_tensor)
        self.robustness_processor = RobustnessProcessor(s=s, d=d, v=v)
        self.merge_processor = MergeProcessor()

    def process_burst(
        self,
        imgs,
        raw_imgs,
        resolution: int = 2,
        better_lk_needed=False,
        verbose=False
    ):
        """

        :param imgs: burst of half-res rgb images
        :param raw_imgs: burst of orig-res raw frames
        :param better_lk_needed: if lk should be counted with blocks (false) or in each point (true)
        :param resolution: int, on which resolution result should be merged (x1 or x2)

        :return: merged frame
        """
        start_time = time.time()
        img_indices = range(len(imgs))
        base_index = 0
        lowres_flows = self.opt_flow_processor.process_alignment(imgs, base_index=base_index, better_lk_needed=False)
        if verbose:
            print("Optical flow took {} seconds".format(time.time() - start_time))
        lowres_kernels = self.kernel_regressor.process_burst(
            imgs, lowres_flows,
            silent=not verbose
        )

        if verbose:
            print("Kernel regression took {} seconds".format(time.time() - start_time))
        start_time = time.time()
        rob_masks = [self.robustness_processor.get_robustness_mask(imgs[0].astype(float) / 255,
                                                    img.astype(float) / 255,
                                                    lowres_flows[index].copy()) for index, img in enumerate(imgs)]

        flows = [upsample_weights(flow, channel=-1) for flow in lowres_flows]
        kernels = [upsample_weights(lowres_kernels[i], channel=-1) for i in range(len(lowres_kernels))]
        masks = [upsample_weights(mask, channel=-1) for mask in rob_masks]
        flows = np.array(flows)
        flows *= 2

        merged = self.merge_processor.merge(raw_imgs, np.array(flows), np.array(kernels), masks, verbose=verbose)

        if resolution == 1:
            return merged

        sparse_raws = [make_sparse_raw(raw) for raw in raw_imgs]

        kernels = self.kernel_regressor.process_burst(
            [merged, *imgs[1:]], flows,
            silent=not verbose,
        )

        if verbose:
            print("Kernel regression took {} seconds".format(time.time() - start_time))

        flows_2x = [upsample_weights(flow, channel=0) for flow in flows]
        kernels_2x = [upsample_weights(kernels[i], channel=-1) for i in range(len(kernels))]
        masks_2x = [upsample_weights(mask, channel=-1) for mask in masks]
        flows_2x = np.array(flows_2x)
        flows_2x *= 2

        merged_2x = self.merge_processor.merge(
            sparse_raws, np.array(flows_2x), np.array(kernels_2x), masks_2x
        )

        if resolution == 2:
            return merged_2x