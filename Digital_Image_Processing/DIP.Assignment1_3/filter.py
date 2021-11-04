import numpy as np

"""
    Module filter include func:
        1. Build filter median -> Reduce noise in image.
        2. Build filter gauss or mean -> Smoothing in image.
        3. Build filter sharp -> Detect edge in image by Laplacian filter.
"""

def gen_gaussian_kernel(k_size: int, sigma: float):
    """
    """
    center = k_size//2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return g


def median_filter(img_np: np.ndarray, filter_size: tuple):
    # First check value in filter size.
    new_img = img_np.copy()
    pad = filter_size[0]//2

    for channel in range(img_np.shape[2]):
        for i in range(pad, img_np.shape[0] - pad):
            for j in range(pad, img_np.shape[1] - pad):
                neighbors = img_np[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
                neighbors.sort()
                new_img[i, j, channel] = neighbors[(filter_size[0]*filter_size[1])//2]
    
    return new_img


def mean_filter(img_np: np.ndarray, filter_size: tuple =(3,3)):
    # Comment after: 
    mean = np.ones((filter_size[0], filter_size[1]), dtype=np.float32)
    pad = filter_size[0]//2
    new_img = img_np.copy()

    for channel in range(img_np.shape[2]):
        for i in range(pad, img_np.shape[0] - pad):
            for j in range(pad, img_np.shape[1] - pad):
                element_wise = img_np[i-pad:i+pad+1, j-pad:j+pad+1, channel]*mean
                new_img[i, j, channel] = int(np.sum(element_wise)/(filter_size[0]*filter_size[1]))

    return new_img
    

def gauss_filter(img_np: np.ndarray, sigma: float, filter_size: tuple =(3, 3)):
    
    gauss = gen_gaussian_kernel(filter_size[0], sigma)
    new_img = img_np.copy()

    pad = filter_size[0]//2
    for channel in range(img_np.shape[2]):
        for i in range(pad, img_np.shape[0] - pad):
            for j in range(pad, img_np.shape[1] - pad):
                element_wise = img_np[i-pad:i+pad+1, j-pad:j+pad+1, channel]*gauss
                new_img[i, j, channel] =  int(np.sum(element_wise))

    return new_img


def laplacian_filter(img_np: np.ndarray):
    laplacian_filter_3x3_v1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_filter_3x3_v2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    new_img = img_np.copy().astype(np.int32)

    for channel in range(img_np.shape[2]):
        for i in range(1, img_np.shape[0] - 1):
            for j in range(1, img_np.shape[1] - 1):
                element_wise = img_np[i-1:i+2, j-1:j+2]*laplacian_filter_3x3_v1
                new_img[i, j, channel] =  np.sum(element_wise)

    return new_img