import numpy as np
from scipy import interpolate
from skimage.exposure import adjust_gamma
    
def upsample_weights(
        lr_weights,
        channel,
):
    """

    :param lr_weights: low resolution weights
    :param channel: position in Bayer pattern
    :return: upsampled weights
    """

    x_shape, y_shape, = lr_weights.shape[0:2]
    #print(x_shape, y_shape)
    x_new = np.mgrid[0:2 * x_shape]
    y_new = np.mgrid[0:2 * y_shape]

    if channel == 0:
        x_old = x_new[::2]
        y_old = y_new[::2]
    elif channel == 1:
        x_old = x_new[1::2]
        y_old = y_new[::2]
    elif channel == 2:
        x_old = x_new[::2]
        y_old = y_new[1::2]
    elif channel == 3:
        x_old = x_new[1::2]
        y_old = y_new[1::2]
    elif channel == -1:        
        x_new = np.mgrid[0:2 * x_shape] + 0.5
        y_new = np.mgrid[0:2 * y_shape] + 0.5
        
        x_old = x_new[::2] + 0.5
        y_old = y_new[::2] + 0.5

    points = np.array([y_old, x_old]).T
    points = (x_old, y_old)
    if channel == -1:
        x_new, y_new = np.mgrid[0.5:2 * x_shape, 0.5:2 * y_shape]
        
    else:
        x_new, y_new = np.mgrid[0:2 * x_shape, 0:2 * y_shape]
    
    new_points = np.array([x_new, y_new]).T
    #print(new_points)
    # new_points = (x_new, y_new)
    z = lr_weights

    new_values = interpolate.interpn(points, z, new_points, method='linear', bounds_error=False)
    new_values = np.nan_to_num(new_values)

    #return new_values
    #print((new_values.T - np.transpose(new_values, axes=(0, 1))).any())
    #return new_values.T
    if len(new_values.shape) == 2:
        return np.transpose(new_values, axes=(1, 0))
    elif len(new_values.shape) == 3:        
        return np.transpose(new_values, axes=(1, 0, 2))
    elif len(new_values.shape) == 4:        
        return np.transpose(new_values, axes=(1, 0, 2, 3))

def postprocess(img, white_balance, color_matrix, white_level):
    img = img * white_balance[:3]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = color_matrix @ img[i][j]
    img = img / white_level
    img[img < 0] = 0
    img[img > 1] = 1
    img = adjust_gamma(img, 1/1.75, 1.5)
#     img = img + white_balance
    return img