import numpy as np
import cv2
from pathlib import Path


def read_file(path: str):
    """

    :param path: Path of file img
    :return: spectrum and angle of image
    """
    assert Path(path).exists(), f"Not exist file {path} in OS."
    assert Path(path).is_file(), f"{path} not type file in OS."

    # Read image source
    img_src = cv2.imread(path)
    # Create image
    h, w = img_src.shape[0], img_src.shape[1]
    img_hsv = np.zeros((h, w, 3))
    img_rgb = np.zeros((h, w, 3))

    # Convert red, green, blue -> hue, saturation, value
    for i in range(h):
        for j in range(w):
            img_hsv[i, j, :] = rgb_to_hsv(img_src[i, j, 2], img_src[i, j, 1], img_src[i, j, 0])
            img_rgb[i, j, :] = hsv_to_rgb(img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2])

    return img_hsv, img_rgb


# Convert rgb to hsv
def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)  # maximum of r, g, b
    cmin = min(r, g, b)  # minimum of r, g, b
    diff = cmax - cmin  # diff of cmax and cmin.

    # Hue calculation
    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # Saturation calculation
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # compute v
    v = cmax * 100
    return h, s/100, v/100


# Convert hsv to rgb
def hsv_to_rgb(h, s, v):
    c = s*v
    x = c*(1-abs((h//60) % 2 - 1))
    m = v - c
    h = int(h//60)

    if h == 0:
        r, g, b = c, x, 0
    elif h == 1:
        r, g, b = x, c, 0
    elif h == 2:
        r, g, b = 0, c, x
    elif h == 3:
        r, g, b = 0, x, c
    elif h == 4:
        r, g, b = x, 0, c
    elif h == 5:
        r, g, b = c, 0, x

    r = int((r + m)*255)
    g = int((g + m)*255)
    b = int((b + m)*255)
    return b, g, r


if __name__ == '__main__':
    img = cv2.imread('C:/Users/Administrator/Documents/Nam4_Ki1/DIP/Digital_Image_Processing/DIP.Assignment1_3/input/Lena.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #img_hsv, img_rgb = read_file('C:/Users/Administrator/Documents/Nam4_Ki1/DIP/Digital_Image_Processing/DIP.Assignment1_3/input/Lena.png')
    cv2.imshow('RGB->HSV', hsv)
   # cv2.imshow('HSV->RGB', img_rgb)
    cv2.waitKey()

