import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from pathlib import Path


def read_file(path: str):
    """

    :param path: Path of file img
    :return: spectrum and angle of image
    """
    assert Path(path).exists(), f"Not exist file {path} in OS."
    assert Path(path).is_file(), f"{path} not type file in OS."

    # Read img
    img = cv2.imread(path, 0)
    return img


# output is a 2D complex array. 1st channel real and 2nd imaginary
# for fft in opencv input image needs to be converted to float32
def DFT(img: np.ndarray):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def ideal_mask_HPF(rows: int, cols: int, r: int):
    return 1 - ideal_mask_LPF(rows, cols, r)


def ideal_mask_LPF(rows: int, cols: int, r: int):
    """

    :param rows: Rows of image
    :param cols: Cols of image
    :param r: radius with center is (0, 0)
    :return: mask LPF
    """
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    return mask


def gaussian_mask_LPF(rows, cols, cutoff):
    """Computes a gaussian low pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the gaussian filter (sigma)
    returns a gaussian low pass mask"""
    d0 = cutoff
    mask = np.zeros((rows, cols, 2))
    mid_R, mid_C = int(rows / 2), int(cols / 2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
            mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))
    return mask


def gaussian_mask_HPF(rows, cols, cutoff):
    # Hint: May be one can use the low pass filter function to get a high pass mask
    d0 = cutoff
    # rows, columns = shape
    # mask = np.zeros((rows, columns), dtype=int)
    mask = 1 - gaussian_mask_LPF(rows, cols, d0)
    return mask


# apply mask and inverse DFT
def inverse_DFT(mask: np.ndarray, dft_shift: np.ndarray):
    fshift = dft_shift * mask
    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img = cv2.idft(f_ishift)
    img = cv2.magnitude(img[:, :, 0], img[:, :, 1])
    return img


def show(src: np.ndarray, dst: np.ndarray, cut: int):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(src, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(dst, cmap='gray')
    ax2.title.set_text('After inverse FFT with cutoff is %d' %cut)
    plt.show()


if __name__ == '__main__':
    src = read_file('C:/Users/Administrator/Documents/Nam4_Ki1/DIP/Digital_Image_Processing/DIP.Assignment1_3/input/Lena.png')
    dft_shift = DFT(src)
    r = 50
    dst = inverse_DFT(gaussian_mask_HPF(src.shape[0], src.shape[1], r), dft_shift)
    show(src, dst, r)
