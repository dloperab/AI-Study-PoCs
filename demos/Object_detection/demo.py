import cv2
import sys
import time
import math
from imutils.video import FPS
from collections import deque
from util import *
import numpy as np
from util import ColorUtilities

cap = cv2.VideoCapture(0)
if not cap.isOpened():
   raise IOError("Cannot open webcam")

pts = deque(maxlen=32)
center = None
mask = None
k=(25,25)
target_bbox = None
fps = None
wname = "frame"
tracker = None

def display_track_path(pts, frame,c=(0,255,0)):
   colors_dict = ColorUtilities.linear_gradient("#1f9623", n=len(pts))
   R = colors_dict["r"]
   G = colors_dict["g"]
   B = colors_dict["b"]
   gradient = list(zip(B,G,R))
   for i in np.arange(1, len(pts)):
      if pts[i - 1] is None or pts[i] is None:
         continue
      thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
      cv2.line(frame, pts[i - 1], pts[i], gradient[i], thickness)

def equalize_rgb_lab(img):
   img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
   clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(4, 4))
   img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
   # convert the YUV image back to RGB format
   img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
   return img_output

def drawRect(img, x, y, w, h, offset=3,c=(0,255,0)):
   # corner top left
   factor = 8
   cv2.line(img, (x + offset, y + offset), (x + offset,
                                             y + int(h / factor) + offset), (0, 255, 0), 2)
   cv2.line(img, (x + offset, y + offset),
            (x + int(w / factor) + offset, y + offset), (0, 255, 0), 2)
   # corner top right
   cv2.line(img, (x + w - offset, y + offset), (x + w -
                                                int(w / factor) - offset, y + offset), (0, 255, 0), 2)
   cv2.line(img, (x + w - offset, y + offset), (x + w - offset,
                                                y + int(h / factor) + offset), (0, 255, 0), 2)
   # bottom top left
   cv2.line(img, (x + offset, y + h - offset), (x + offset,
                                                y + h - int(h / factor) - offset), (0, 255, 0), 2)
   cv2.line(img, (x + offset, y + h - offset),
            (x + int(w / factor) + offset, y + h - offset), (0, 255, 0), 2)
   # bottom top right
   cv2.line(img, (x + w - offset, y + h - offset), (x + w - \
            int(w / factor) - offset, y + h - offset), (0, 255, 0), 2)
   cv2.line(img, (x + w - offset, y + h - offset), (x + w - \
            offset, y + h - int(h / factor) - offset), (0, 255, 0), 2)

def drawCenter(img, x, y, w, h,c=(0,255,0)):
   # draw origin
   origin = (math.floor(x + w / 2), math.floor(y + h / 2))
   cv2.circle(img, origin, 2, c,-1 )
   drawline(
      img,
      (origin[0] - 20,
         origin[1]),
      (origin[0] + 21,
         origin[1]),
      c,
      thickness=1,
      style='dotted',
      gap=5)
   drawline(
      img,
      (origin[0],
         origin[1] - 20),
      (origin[0],
         origin[1] + 21),
      c,
      thickness=1,
      style='dotted',
      gap=5)

def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
   dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
   pts = []
   for i in np.arange(0, dist, gap):
      r = i / dist
      x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
      y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
      p = (x, y)
      pts.append(p)
   if style == 'dotted':
      for p in pts:
            cv2.circle(img, p, thickness, color, -1)
   else:
      s = pts[0]
      e = pts[0]
      i = 0
      for p in pts:
            s = e
            e = p
            if i % 2 == 1:
               cv2.line(img, s, e, color, thickness)
            i += 1
   
def drawGrid(img,x, y, w, h, steps=40, alpha=0.2,c=(0,255,0)):        
     
      roi = img[y:y+h,x:x+w].copy()
      overlay = np.zeros(roi.shape, dtype=roi.dtype)
      # vertical lines
      vlines = np.arange(0, overlay.shape[0], steps)
      # horizontal lines
      hlines = np.arange(0, overlay.shape[1], steps)

      # draw lines        
      for i in hlines:
         cv2.line(overlay, (i, 0), (i, overlay.shape[0]), (211, 211, 211), 1)
      for j in vlines:
         cv2.line(overlay, (0, j), (overlay.shape[1], j), (211, 211, 211), 1)

      roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)            
      
      img[y:y+h,x:x+w] = roi


while True:
   try:
      ret, frame = cap.read()
      '''frame = cv2.resize(
         frame, 
         None, 
         fx=0.8, 
         fy=0.8, 
         interpolation=cv2.INTER_AREA)'''
      frame = cv2.flip(frame, 1)
      frame = equalize_rgb_lab(frame)
      mask = np.zeros(frame.shape, dtype=frame.dtype)
      if target_bbox is not None:
      # grab the new bounding box coordinates of the object
         (success, box) = tracker.update(frame)
         # check to see if the tracking was a success
         if success:
            (H,W,_) = frame.shape
            (x, y, w, h) = [int(v) for v in box]
            center = (math.floor(x+w/2),math.floor(y+h/2))
            pts.appendleft(center)
            #cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 1)
            drawRect(frame,x, y,w, h, offset=8)
            cv2.circle(frame, center, 2, (0,255,0),-1 )
            #create mask            
            mask[y:y+h,x:x+w] = 255
            drawCenter(frame,x,y,w,h)
            drawGrid(frame,x,y,w,h)
            roi = frame[y:y+h,x:x+w].copy()
            frame = cv2.GaussianBlur(frame, k, 0)
            frame[y:y+h,x:x+w] = roi
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            #print("FPS", "{:.2f}".format(fps.fps()))
            cv2.putText(frame, 
            "FPS: {:.2f}".format(fps.fps()), 
            (10, H - ((0 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1)

            display_track_path(pts, frame)
         else:
            target_bbox = None   
      vis = np.concatenate((frame, mask), axis=1)
      cv2.imshow(wname, vis)
      
      key = cv2.waitKey(1) % 255
      if key == ord("1"):
         target_bbox = cv2.selectROI(
            wname, 
            frame, 
            fromCenter=False,
            showCrosshair=True)
         pts = deque(maxlen=32)
         tracker = cv2.TrackerMOSSE_create()
         tracker.init(frame, target_bbox)
         fps = FPS().start()
      if key == ord("2"):
         target_bbox = None
      elif key == ord("q"):
         break
   except Exception as ex:
      print(ex)
      
cap.release()
cv2.destroyAllWindows()
