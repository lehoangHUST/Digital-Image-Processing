import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from pathlib import Path


def read_file(path: str):
    """

    :param path: Path of file img
    :return: spectrum and angle of image
    """
    assert Path(path).exists(), f"Not exist file {path} in OS."
    assert Path(path).is_file(), f"{path} not type file in OS."

    # Read img
    img = cv.imread(path, 0)

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # Spectrum and angle
    spectrum, angle = 200*np.log(cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    print("========  Loading image %s ========" %path)
    show(spectrum, angle, path)


def show(spectrum: np.ndarray, angle: np.ndarray, path: str):
    print(spectrum)
    print(angle)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(spectrum, cmap='gray')
    ax1.title.set_text('Spectrum')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(angle, cmap='gray')
    ax2.title.set_text('Angle')
    plt.show()


def read_folder(path: str):
    """
    :param path: List file in folder
    :return: Nothing
    """
    assert Path(path).exists(), f"Not exist folder {path} in OS."
    assert Path(path).is_dir(), f"{path} not type folder in OS."

    # List file in folder path
    abs_path = Path(path).absolute()
    for file in os.listdir(abs_path):
        f = os.path.join(abs_path, file)
        read_file(f)


if __name__ == '__main__':
    #read_folder('./Endo')
    read_file('./Endo/HMUH_01 201007_200226_BN103_008.jpg')

