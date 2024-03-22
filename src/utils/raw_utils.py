import numpy as np

def make_sparse_raw(raw):
    """

    :param raw: mosaic (h, w, 3)
    :return: also mosaic but sparse, shape (2h, 2w, 3)
    """

    sparse_raw = np.zeros((raw.shape[0] * 2, raw.shape[1] * 2, 3), dtype=float)

    if raw[::2, ::2, 1].any():
        if raw[::2,1::2,2].any():
            sparse_raw[::4, ::4, 1] = raw[::2, ::2, 1]
            sparse_raw[::4, 2::4, 2] = raw[::2, 1::2, 2]
            sparse_raw[2::4, ::4, 0] = raw[1::2, ::2, 0]
            sparse_raw[2::4, 2::4, 1] = raw[1::2, 1::2, 1]
        else:
            sparse_raw[::4, ::4, 1] = raw[::2, ::2, 1]
            sparse_raw[::4, 2::4, 0] = raw[::2, 1::2, 0]
            sparse_raw[2::4, ::4, 2] = raw[1::2, ::2, 2]
            sparse_raw[2::4, 2::4, 1] = raw[1::2, 1::2, 1]
            
    elif raw[::2, ::2, 0].any():
        sparse_raw[::4, ::4, 0] = raw[::2, ::2, 0]
        sparse_raw[::4, 2::4, 1] = raw[::2, 1::2, 1]
        sparse_raw[2::4, ::4, 1] = raw[1::2, ::2, 1]
        sparse_raw[2::4, 2::4, 2] = raw[1::2, 1::2, 2]
    elif raw[::2, ::2, 2].any():
        sparse_raw[::4, ::4, 2] = raw[::2, ::2, 2]
        sparse_raw[::4, 2::4, 1] = raw[::2, 1::2, 1]
        sparse_raw[2::4, ::4, 1] = raw[1::2, ::2, 1]
        sparse_raw[2::4, 2::4, 0] = raw[1::2, 1::2, 0]

    return sparse_raw