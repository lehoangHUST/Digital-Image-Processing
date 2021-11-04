import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import argparse

from filter import *
from histogram import *

# Add argument from command line or cmd in Linux or Windows 10.
def add_argument():
  # Init add argument from command line
  parser = argparse.ArgumentParser(
    description='Add Argument for task in image.')
  
  # Add argument.
  parser.add_argument('--input', default=None, type=str,
                      help='Image input is path file and have type string.')
  parser.add_argument('--convert_RGB', default=True, type=int,
                      help='We can choose image convert RGB or Gray. If RGB => True, else is False.')
  parser.add_argument('--save_info_image', default=True, type=bool,
                      help='Save all information in image. Example: dynamic range, histogram, height and width of image_input')
  parser.add_argument('--case', default=None, type=int,
                      help='Choose case to implement task.')
  
  args = parser.parse_args()
  return args

# FORMATS OF IMAGE AND VIDEO
IMG_FORMATS = ['jpg', 'jpeg', 'png']  # acceptable image suffixes
VID_FORMATS = ['avi', 'mp4', 'gif']  # acceptable video suffixes

# TASK CASE IN ASSIGNMENT.
CASE = ['output',
        'draw_hist',
        'scale_histogram',
        'histogram_equalizion',
        'Median filter',
        'Mean filter',
        'Gauss filter',
        'Sharp filter'] 

# '' mean nothing task.

# Built class Image
class Image:
  """
      We built class Image include func:
        1. Calculate cumdf histogram of image. Example: Image gray, Image RGB or BGR....
        2. Calculate dynamic range of image. Example: Image have to intensity from min: 0 -> max: 156.
        3. Write information all about image in json.
  """

  def __init__(self, path_file: str, RGB: int):
    # Func init paras input
    """
      + Paras: 
          1. Para1: path_img is path of image in system. Maybe path_img is folder or file, self.path_file .
          2. Para2: RGB is convert image from RGB or convert Gray, self.RGB .    
    """
    self.path_file = path_file
    self.RGB = RGB
    self.name_img = path_file.split('/')[-1].split('.')[0]

    # Check paras is true type ??
    assert os.path.isfile(self.path_file), f"Not file image {self.path_file} in system."
    assert isinstance(self.RGB, int), f"Para self.RGB only support {type(self.RGB)}."

    self.img_np = cv2.imread(self.path_file, self.RGB)
    if self.RGB == 0:
      self.img_np = self.img_np.reshape((self.img_np.shape[0], self.img_np.shape[1], 1))


  def dynamic_range(self):
    # calculate the dynamic range of values in that picture
    if self.img_np.shape[2] == 3:
      dynamic = {'Red': [np.min(self.img_np[:, : , 0]), np.max(self.img_np[:, :, 0])] ,
                'Green': [np.min(self.img_np[:, :, 1]), np.max(self.img_np[:, :, 1])],
                'Blue': [np.min(self.img_np[:, :, 2]), np.max(self.img_np[:, :, 2])]
                }
    else:
      dynamic = {'Gray': [np.min(self.img_np[:, :]), np.max(self.img_np[:, :])]}
      # Each channels Red, Blue or Green have dynamic range ?
      # Specific convert image have range intensity [L1, L2] to [L1', L2'].
    
    return dynamic
  
  # 4. Save file json (Contains about infor image.)
  def save_folder(self):
    print('*'*30)
    print("Start .... Loading")
    # Information
    infor_img = {
      'path': [],
      'dynamic range': [],
      'dimensional': []
    }
    infor_img['path'].append(self.path_file)
    infor_img['dynamic range'].append(self.dynamic_range())
    infor_img['dimensional'].append([self.img_np.shape[0], self.img_np.shape[1]])

    print(infor_img)
    # Write infor in file .json
    # with open(self.name_img + '_output' + '.json', 'w') as f:
    #  json.dump(infor_img, f)
    print("Finish")
    print('*'*30)


# Run main
def run():

  # Read image if args.input have suffixes is '.jpg', '.jpeg', '.png'
  # Read video if args.input have suffixes is '.mp4', '.avi', '.gif'

  if args.input.split('/')[-1].split('.')[1] in IMG_FORMATS:
    img = Image(args.input, args.convert_RGB)
  else:
    raise ValueError(f"Parameters not support for {VID_FORMATS}")

  # Important
  """
      case 0: Input: Image -> Output: cv2.imread() -> Type np.ndarray.
      case 1: Input: Image, bin -> Output: List histogram of image. 
      case 2: Input: Image, src_range, dst_range -> Output: Image use scale histogram (Linear Transform)
      case 3: Input: Image -> Output: Image with Histogram Equalizion
      case 4: Input: Image -> Output: Image after use median filter.
      case 5: Input: Image -> Output: Image after use smoothing filter (Mean Filter)
      case 6: Input: Image -> Output: Image after use smoothing filter (Gauss Filter)
      case 7: Input: Image -> Output: Image after use sharpening filter (Laplacian filter)
  """
  case = args.case
  # All case from 0 -> 6.
  if case == 0:
    # Write image and show image in display.
    cv2.imwrite(img.name_img + '_output' + '.png', img.img_np)
  elif case == 1:
    # Count histogram.
    hist = count_histogram(img.img_np)
    intensity_img = [x for x in range(256)]
    # Save plot of histogram in intensity.
    for channel in range(img.img_np.shape[2]):
      plt.plot(intensity_img, hist['Channel '+ str(channel + 1)])
      plt.xlabel('Intensity of image')
      plt.ylabel('Histogram of intensity')
      plt.savefig('Channel '+ str(channel + 1) + '.png')
      plt.title('Channel '+ str(channel + 1))
      plt.clf()
  elif case == 2:
    # Use for one channel is: Gray.
    src = img.dynamic_range()['Gray']
    new_img = scale_histogram(img.img_np, src, dst_range = [0, 200])
    cv2.imwrite(img.name_img + '_scale_histogram' + '.png', new_img)
  elif case == 3:
    new_img = histogram_equalizion(img.img_np)
    cv2.imwrite(img.name_img + '_histogram_equal' + '.png', new_img)
  elif case == 4:
    # Use median filter to reduce noise
    new_img = median_filter(img.img_np, filter_size=(5, 5))
    cv2.imwrite(img.name_img + '_median' + '.png', new_img)
  elif case == 5:
    new_img = mean_filter(img.img_np, filter_size=(5, 5))
    cv2.imwrite(img.name_img + '_mean' + '.png', new_img)
  elif case == 6:
    new_img = gauss_filter(img.img_np, filter_size=(5, 5), sigma=1)
    cv2.imwrite(img.name_img + '_gauss' + '.png', new_img)
  elif case == 7:
    new_img = laplacian_filter(img.img_np)
    cv2.imwrite(img.name_img + '_sharp' + '.png', new_img)
  else:
    raise ValueError("Not found case : {case} in program.")

# Run main
if __name__ == '__main__':
  global args
  args = add_argument()
  run()