import cv2
import numpy as np
import argparse
import datetime
import os

_DISPLAY_NAME = 'ra_detector'

def GetArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame_mod', dest='frame_mod', action='store', default=1, type=int,
    help='Process every [frame_mod] frames')
  parser.add_argument('--output_folder', dest='output_folder', default='', action='store', 
    help='If supplied, write output images to this folder')
  return parser.parse_args()


def PreprocessImage(image):
  """Preprocess the image by applying a blur"""
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = image.astype(float)
  image = cv2.GaussianBlur(image, (7,7), 1)
  return image
  
def ComputeSOD(frame1, frame2):
  """Compute sum of difference between frames
  Applies a filter on the difference image to reduce noise: 
  https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  """
  if frame1 is None or frame2 is None:
    return 0
  else:
    diff = np.abs(frame2 - frame1)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((11,11)))
    return np.abs(np.sum(diff.flatten())), diff
    
def GetContours(image):
  """Find contours(connected segments) in an image"""
  image = image.astype(np.uint8)
  # Threshold the image about a certain value.
  ret, image = cv2.threshold(image, 15, 255, 0)
  return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def GetDebugImage(contours, image):
  """Draws bounding boxes around contours for debugging"""
  debug_image = np.copy(image)
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(debug_image, (x,y), (x+w, y+h), (0,0,255), 2)
  return debug_image 
  
    
def main():
  args = GetArgs()
  frame_count = 0 
  cv2.namedWindow(_DISPLAY_NAME)

  video = cv2.VideoCapture(0)
  prev_frame = None
  while True:
    valid, img = video.read()
    if valid:
      frame_count += 1
      if frame_count % args.frame_mod == 0:
        current_frame = (img, PreprocessImage(img))
        if prev_frame is not None:
          sod, diff_image = ComputeSOD(prev_frame[1], current_frame[1])
          contours, _ = GetContours(diff_image)
          debug_image = GetDebugImage(contours, img)         
          if len(contours) > 0:
            print('MOVEMENT!') 
            if len(args.output_folder) > 0:
              now = datetime.datetime.now()    
              date_string = now.strftime('%Y-%m-%d_%H-%M-%S.%f')
              cv2.imwrite(os.path.join(args.output_folder, '%s.png' % date_string), img)

          cv2.imshow(_DISPLAY_NAME, debug_image) 
          cv2.waitKey(1)

        prev_frame = current_frame

if __name__ == '__main__':
  main()
