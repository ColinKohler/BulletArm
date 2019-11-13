import numpy as np
from scipy import ndimage

def rotateImage(img, angle, pivot):
  pad_x = [img.shape[1] - pivot[1], pivot[1]]
  pad_y = [img.shape[0] - pivot[0], pivot[0]]
  img_recenter = np.pad(img, [pad_y, pad_x], 'constant')
  img_2x = ndimage.zoom(img_recenter, zoom=2, order=0)
  img_r_2x = ndimage.rotate(img_2x, angle, reshape=False)
  img_r = img_r_2x[::2, ::2]
  if pad_y[1] == 0 and pad_x[1]== 0:
    result = img_r[pad_y[0]:, pad_x[0]:]
  elif pad_y[1] == 0:
    result = img_r[pad_y[0]:, pad_x[0]: -pad_x[1]]
  elif pad_x[1] == 0:
    result = img_r[pad_y[0]: -pad_y[1], pad_x[0]:]
  else:
    result = img_r[pad_y[0]: -pad_y[1], pad_x[0]: -pad_x[1]]
  return result
