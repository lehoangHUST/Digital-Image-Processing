import cv2
import numpy as np

# Convert .gif -> numpy
def video_to_numpy(path: str, convert_RGB: int):
    # Read file .gif
    gif = cv2.VideoCapture(path, convert_RGB)

    # Init frame = 0
    frame_num = 0
    img_list = []

    # Loop until not read frame
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not.
            okay, frame = gif.read()
            # Apppend list
            img_list.append(frame)

            # Break if there are no other frames to read
            if not okay:
                break
            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

    return np.array(img_list)