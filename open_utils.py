import rawpy
import skimage

import numpy as np

from skimage import transform
from pathlib import Path


def open_raw(filename: str, raw_needed: bool = False, half_size: bool = True):
    """
    :param filename:
    :param raw_needed: if True, return raw mosaiced image
    :return: (RAW or RGB image, info about raw image)
    """
    img_raw = rawpy.imread(filename)
    white_balance = img_raw.daylight_whitebalance
    color_matrix = img_raw.color_matrix[:, :3]
    white_level = img_raw.white_level
    if raw_needed:
        raw_image = img_raw.raw_image.astype(float)
        black_offset = img_raw.black_level_per_channel
        top_margin = img_raw.sizes.top_margin
        left_margin = img_raw.sizes.left_margin
        iheight = img_raw.sizes.iheight
        iwidth = img_raw.sizes.iwidth

        img = np.zeros((iheight, iwidth, 3))
        img[::2, ::2, 2] = (
            raw_image[
                top_margin : top_margin + iheight : 2,
                left_margin : left_margin + iwidth : 2,
            ]
            - black_offset[0]
        )
        img[1::2, ::2, 1] = (
            raw_image[
                1 + top_margin : top_margin + iheight : 2,
                left_margin : left_margin + iwidth : 2,
            ]
            - black_offset[1]
        )
        img[::2, 1::2, 1] = (
            raw_image[
                top_margin : top_margin + iheight : 2,
                1 + left_margin : left_margin + iwidth : 2,
            ]
            - black_offset[3]
        )
        img[1::2, 1::2, 0] = (
            raw_image[
                1 + top_margin : top_margin + iheight : 2,
                1 + left_margin : left_margin + iwidth : 2,
            ]
            - black_offset[2]
        )

        img_rgb = img
        if img_raw.sizes.flip == 6:
            img_rgb = skimage.transform.rotate(
                img_rgb, angle=270, resize=True
            )  # only for example
        img_rgb[img_rgb < 0.01] = 0
    else:
        img_rgb = img_raw.postprocess(half_size=half_size)
    info = {
        "white_balance": white_balance,
        "color_matrix": color_matrix,
        "white_level": white_level,
    }
    return img_rgb, info


def open_burst_of_frames(
    directory,
    img_indices=None,
    format=".dng",
    up=0,
    down=-1,
    left=0,
    right=-1,
    raw_needed=False,
):
    """
    Opens a directory with RAW images and returns a list of them, filtering by indices if needed

    :param directory: directory with RAW images
    :param img_indices: indices of images to open
    :param format: format of images
    :param up, dowm, left, right: cropping parametes (may be needed to reduce working time)
    :param raw_needed: if True, return raw mosaiced image

    :return: list of images, info about raw images
    """
    imgs = []
    infos = []

    if raw_needed:
        up *= 2
        down *= 2
        left *= 2
        right *= 2

    for i, filename in enumerate(Path(directory).glob(f"*{format}")):
        if img_indices is not None and i not in img_indices:
            continue
        img, info = open_raw(str(filename), raw_needed=raw_needed)
        imgs.append(img[up:down, left:right])
        infos.append(info)
    return imgs, infos
