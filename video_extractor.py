import os
import numpy as np

import cv2
vidcap = cv2.VideoCapture('20190821_201113.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/frame%0000d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
