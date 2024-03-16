from alignment_utils import calc_d2, downsample, extract
import cv2
import time
import skimage
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import RectBivariateSpline
from skimage.color import rgb2gray
from skimage import measure


class OpticalFlowProcessor:
    def __init__(
            self,
            depth=4,
            size=16,
            k_iters=3,
            var_threshold=8.0,
            threshold_for_lk: float=0.1,
            ):
        self.depth = depth
        self.size = size
        self.tile_size = size # TODO: change to size
        self.k_iters = k_iters
        self.var_threshold = var_threshold
        self.threshold_for_lk = threshold_for_lk

    def align_template(self, template, extended, center, debug_verbose=False):
        """
        Finds the best match between a template and an extended image (shift with highest correlation)

        template: template image to find in extended
        extended: second image, in which template is searched
        center: center of template

        return: best integer shift to find template in extended
        """
        d2_distance = calc_d2(template, extended)
        argmax = np.unravel_index(np.argmin(d2_distance, axis=None), d2_distance.shape)
        argmax = np.array(argmax)
        shift = argmax - center
        if debug_verbose:
            print(argmax, shift, d2_distance.max())
        return shift


    def align_block_based(self, base, image, depth=4, verbose=False):
        """
        Computes block-based optical flow between two images with recursive calls to downsampled versions of images

        base: first image
        image: second image
        depth: depth of recursion
        tile_size: size of blocks

        return: optical flow with integer shifts
        """
        if depth == 1:
            if verbose:
                print(f"processing image at depth = 1")
            n_horizontal_blocks = base.shape[1] // self.tile_size + 1
            n_vertical_blocks = base.shape[0] // self.tile_size + 1
            shifts = np.zeros((base.shape[0], base.shape[1], 2), np.float32)
            for i in range(0, n_vertical_blocks):
                if i < n_vertical_blocks - 1:
                    up = i * self.tile_size
                else:
                    up = base.shape[0] - self.tile_size
                down = up + self.tile_size

                for j in range(0, n_horizontal_blocks):
                    if j < n_horizontal_blocks - 1:
                        left = j * self.tile_size
                    else:
                        left = base.shape[1] - self.tile_size
                    right = left + self.tile_size

                    template = base[up:down, left:right]

                    if np.var(rgb2gray(template)) > self.var_threshold:
                        shifts[up:down, left:right] = np.zeros(2)
                        continue

                    search_area = image[
                        max(up - 4, 0) : min(down + 4, image.shape[0]),
                        max(left - 4, 0) : min(right + 4, image.shape[1]),
                    ]

                    if up - 4 < 0:
                        center_y = 8
                    else:
                        center_y = self.tile_size // 2 + 4
                    if left - 4 < 0:
                        center_x = 8
                    else:
                        center_x = self.tile_size // 2 + 4
                    center = np.array([center_y, center_x])

                    shift = self.align_template(template, search_area, center)
                    shifts[up:down, left:right] = shift
            return shifts
        else:
            base_downsampled = downsample(base)
            image_downsampled = downsample(image)
            shifts_downsampled = self.align_block_based(
                base_downsampled, image_downsampled, depth - 1
            )
            if verbose:
                print(f"processing image at depth = {depth}")
            low_shape = shifts_downsampled.shape
            shifts_upsampled = 2 * cv2.resize(
                shifts_downsampled,
                dsize=(2 * low_shape[1], 2 * low_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            n_horizontal_blocks = base.shape[1] // self.tile_size + 1
            n_vertical_blocks = base.shape[0] // self.tile_size + 1
            shifts = np.zeros((base.shape[0], base.shape[1], 2))
            for i in range(0, n_vertical_blocks):
                if i < n_vertical_blocks - 1:
                    up = i * self.tile_size
                else:
                    up = base.shape[0] - self.tile_size
                down = up + self.tile_size

                for j in range(0, n_horizontal_blocks):
                    if j < n_horizontal_blocks - 1:
                        left = j * self.tile_size
                    else:
                        left = base.shape[1] - self.tile_size
                    right = left + self.tile_size

                    template = base[up:down, left:right]
                    if np.var(rgb2gray(template)) < self.var_threshold:
                        shifts[up:down, left:right] = np.zeros(2)
                        continue
                    lower_shift = shifts_upsampled[up][left]
                    search_area = image[
                        max(up + int(lower_shift[0]) - 4, 0) : min(
                            down + int(lower_shift[0]) + 4, image.shape[0]
                        ),
                        max(left + int(lower_shift[1]) - 4, 0) : min(
                            right + int(lower_shift[1]) + 4, image.shape[1]
                        ),
                    ]

                    if up + lower_shift[0] - 4 < 0:
                        center_y = self.tile_size // 2 + (up + lower_shift[0] - 4)
                    else:
                        center_y = self.tile_size // 2 + 4
                    if left + lower_shift[1] - 4 < 0:
                        center_x = self.tile_size // 2 + (left + lower_shift[1] - 4)
                    else:
                        center_x = self.tile_size // 2 + 4
                    center = np.array([center_y, center_x])
                    shift = self.align_template(template, search_area, center)
                    shifts[up:down, left:right] = shift + lower_shift

            shifts[0:self.tile_size, :] = shifts[self.tile_size, :]
            shifts[shifts.shape[0] - self.tile_size : shifts.shape[0], :] = shifts[
                shifts.shape[0] - self.tile_size - 1
            ]

            for i in range(self.tile_size):
                shifts[:, i] = shifts[:, self.tile_size]
                shifts[:, shifts.shape[1] - i - 1] = shifts[
                    :, shifts.shape[1] - self.tile_size - 1
                ]

            return shifts


    def LucasKanade(self, It, It1, rect, p0=np.zeros(2), threshold=0.1):
        """
        Lucas-Kanade method for optical flow estimation

        :param It: template image
        :param It1: Current image
        :param rect: Current position of the car
        :param p0: Initial movement vector [dp_x0, dp_y0]
        :param threshold:
        :return: p: movement vector [dp_x, dp_y]


        o____x
        |
        |
        y image(y, x) opencv convention
        """

        threshold = threshold
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        rows_img, cols_img = It.shape
        rows_rect, cols_rect = x2 - x1, y2 - y1
        dp = [[cols_img], [rows_img]]  # just an intial value to enforce the loop

        # template-related can be precomputed
        Iy, Ix = np.gradient(It1)
        y = np.arange(0, rows_img, 1)
        x = np.arange(0, cols_img, 1)
        c = np.linspace(x1, x2, cols_rect)
        r = np.linspace(y1, y2, rows_rect)
        cc, rr = np.meshgrid(c, r)
        spline = RectBivariateSpline(y, x, It)
        T = spline.ev(rr, cc)
        spline_gx = RectBivariateSpline(y, x, Ix)
        spline_gy = RectBivariateSpline(y, x, Iy)
        spline1 = RectBivariateSpline(y, x, It1)

        # in translation model jacobian is not related to coordinates
        jac = np.array([[1, 0], [0, 1]])
        iter_num = 0

        while np.square(dp).sum() > threshold and iter_num < 3:
            iter_num += 1
            # warp image using translation motion model
            x1_w, y1_w, x2_w, y2_w = x1 + p0[0], y1 + p0[1], x2 + p0[0], y2 + p0[1]

            cw = np.linspace(x1_w, x2_w, cols_rect)
            rw = np.linspace(y1_w, y2_w, rows_rect)
            ccw, rrw = np.meshgrid(cw, rw)

            warpImg = spline1.ev(rrw, ccw)

            # compute error image
            err = T - warpImg
            errImg = err.reshape(-1, 1)

            # compute gradient
            Ix_w = spline_gx.ev(rrw, ccw)
            Iy_w = spline_gy.ev(rrw, ccw)
            # I is (n,2)
            I = np.vstack((Ix_w.ravel(), Iy_w.ravel())).T

            # computer Hessian
            delta = I @ jac
            # H is (2,2)
            H = delta.T @ delta

            # compute dp
            # dp is (2,2)@(2,n)@(n,1) = (2,1)
            try:
                dp = np.linalg.inv(H) @ (delta.T) @ errImg
            except np.linalg.LinAlgError:
                break
                robust = 0.01
                dp = np.linalg.inv(H + robust * np.eye(H.shape[0])) @ (delta.T) @ errImg

            # update parameters
            p0[0] += dp[0, 0]
            p0[1] += dp[1, 0]

        return p0


    def LK_after_BB(self, base_frame, image, flow):
        """
        Applies LK after block-based optical flow estimation
        NB: We can't use LK without coarse optical flow estimation
        because LK method is based on Taylor series and its robustness requires small shifts

        :param base_frame: base frame
        :param image: frame to be aligned
        :param flow: coarse flow estimation with integer shifts
        :param tile_size: tile size
        :param k_iters: number of iterations
        :param threshold_for_lk: threshold for LK

        :return: optical flow refined with integer shifts
        """
        var_treshold = 7.0
        n_horizontal_blocks = base_frame.shape[1] // self.tile_size + 1
        n_vertical_blocks = base_frame.shape[0] // self.tile_size + 1
        shifts = np.zeros((base_frame.shape[0], base_frame.shape[1], 2), dtype=np.float32)
        for i in range(0, n_vertical_blocks):
            if i < n_vertical_blocks - 1:
                up = i * self.tile_size
            else:
                up = base_frame.shape[0] - self.tile_size
            down = up + self.tile_size

            for j in range(0, n_horizontal_blocks):
                if j < n_horizontal_blocks - 1:
                    left = j * self.tile_size
                else:
                    left = base_frame.shape[1] - self.tile_size
                right = left + self.tile_size

                template, shifted = extract(
                    base_frame, image, flow[up][left], i, j, block_size=self.tile_size
                )
                if (
                    template is None
                    or j in [0, n_horizontal_blocks - 1]
                    or i in [0, n_vertical_blocks - 1]
                ):
                    point_flow = np.zeros(2)
                else:
                    point_flow = np.zeros(2)
                    if np.var(rgb2gray(template)) > var_treshold:
                        for k in range(self.k_iters):
                            point_flow = self.LucasKanade(
                                rgb2gray(template),
                                rgb2gray(shifted),
                                (0, 0, template.shape[0], template.shape[1]),
                                p0=point_flow,
                                threshold=self.threshold_for_lk,
                            )
                shifts[up:down, left:right] = point_flow

        return shifts


    def get_lk_optical_flow(
            self,
        base,
        to_align,
        better_lk_needed=False,
        verbose=False,
    ):
        # size = 16
        start_time = time.time()
        flow_bb = self.align_block_based(
            base.astype(np.float32), to_align.astype(np.float32), verbose=verbose
        )

        flow_bb = flow_bb[
            :, :, ::-1
        ]  # before: flow = [flow_y, flow_x], after: flow = [flow_x, flow_y]
        # 0_x
        # |
        # y

        flow_bb = cv2.copyMakeBorder(
            flow_bb[self.size:-self.size, self.size:-self.size], self.size, self.size, self.size, self.size, cv2.BORDER_REPLICATE
        )

        if better_lk_needed:
            flow_LK = self.better_LK(base, to_align, flow_bb.copy())
        else:
            flow_LK = self.LK_after_BB(
                base.astype(np.float32),
                to_align.astype(np.float32),
                flow_bb.copy(),
            )

        print(f"alignment for shape {base.shape} taken {time.time() - start_time} seconds")
        return flow_bb, flow_LK


    def better_LK(self, base_frame, image, flow):
        base_frame = rgb2gray(base_frame)
        image = rgb2gray(image)
        flow = flow.astype(int)
        var_treshold = 10.0
        shifts = np.zeros((base_frame.shape[0], base_frame.shape[1], 2))

        for row in range(self.size, base_frame.shape[0] - self.size):
            for col in range(self.size, base_frame.shape[1] - self.size):
                if row % 3 != 1 or col % 3 != 1:
                    continue
                tpl = base_frame[
                    row - self.size // 2 : row + self.size // 2 + 1,
                    col - self.size // 2 : col + self.size // 2 + 1,
                ]

                up = max(row - self.size // 2 + flow[row][col][1], 0)
                down = min(row + self.size // 2 + 1 + flow[row][col][1], base_frame.shape[0])
                left = max(col - self.size // 2 + flow[row][col][0], 0)
                right = min(col + self.size // 2 + 1 + flow[row][col][0], base_frame.shape[1])
                img_crop = image[up:down, left:right]

                point_flow = np.zeros(2)
                if np.var(tpl) > var_treshold and tpl.shape == img_crop.shape:
                    for k in range(3):
                        point_flow = self.LucasKanade(
                            tpl, img_crop, (0, 0, tpl.shape[0], tpl.shape[1]), p0=point_flow
                        )

                shifts[row - 1 : row + 2, col - 1 : col + 2] = point_flow
        shifts = cv2.copyMakeBorder(
            shifts[self.size:-self.size, self.size:-self.size], self.size, self.size, self.size, self.size, cv2.BORDER_REPLICATE
        )
        return shifts


    def process_alignment(
        self, imgs, base_index=0, better_lk_needed=False, verbose=False
    ):
        # TODO: remove outdated img_indices
        """
        flow_bb - block_based
        flow_lk - Lucas-Kanade

        :param imgs: list of images
        :param base_index:
        :return: list of optical flows
        """
        flows = []
        for img_i, img in enumerate(imgs):
            flow_bb, flow_lk = self.get_lk_optical_flow(
                imgs[base_index].astype(float),
                img.astype(float),
                better_lk_needed=better_lk_needed,
                verbose=verbose,
            )
            flows.append(flow_bb + flow_lk)

        return flows
