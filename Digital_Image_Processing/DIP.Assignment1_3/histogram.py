import cv2
import numpy as np
from itertools import product 


# Count histogram.
def count_histogram(img_np: np.ndarray):
    hist = np.zeros((img_np.shape[2], 256)) # If have 256 intensity histogram by image 8-bit.
    img_hist = {}
    for channel in range(img_np.shape[2]):
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                hist[channel, img_np[i, j, channel]] += 1 

        img_hist['Channel ' + str(channel + 1)] = hist[channel]

    return hist

# Contrast strech
# Convert contrast intensity [L1, L2] to [L1', L2']
def scale_histogram(img_np: np.ndarray, src_range: list, dst_range: list):

    # Change source intensity range to dst intensity range.
    # Example: [0, 5] -> [0, 10].
    # { 0 = 0a + b  -> { a = 2
    # {10 = 5a + b  -> { b = 0
    # Specific : Find value a and b base on [L1, L2] and [L1', L2']
    # a = (L2' - L1')/(L2 - L1); b = L1' - L1*(L2' - L1')/(L2 - L1)
    Bias_1 = (dst_range[1] - dst_range[0])/(src_range[1] - src_range[0])
    Bias_2 = dst_range[0] - src_range[0]*Bias_1

    new_img = img_np.copy() # Copy

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            new_img[i, j] = int(Bias_1*img_np[i,j] + Bias_2)

    return new_img

# Histogram Equalizion.
def histogram_equalizion(img_np: np.ndarray):
    # Default range intensity [L1, L2] -> [0, 255] -> 8 bit
    mean_intensity = (img_np.shape[0]*img_np.shape[1])//256
    print(mean_intensity)
    # Count hist 
    _hist_ = count_histogram(img_np)
    cumdf = np.zeros((_hist_.shape[0], 256))
    new_hist = _hist_.copy()

    # Change
    for channel in range(_hist_.shape[0]):
        for intensity in range(256):
            if intensity == 0:
                cumdf[channel, intensity] = _hist_[channel, intensity]
            else:
                cumdf[channel, intensity] = cumdf[channel, intensity - 1] + _hist_[channel, intensity]

            new_hist[channel, intensity] = max(0, int(np.round(cumdf[channel, intensity]/mean_intensity)) - 1)

    print(new_hist.shape)
    # New img
    new_img = img_np.copy()
    for channel in range(img_np.shape[2]):
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                new_img[i, j, channel] = new_hist[channel, [img_np[i, j, channel]]]
    
    return new_img