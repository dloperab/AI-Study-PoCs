import sys
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1,parentdir)
import cv2
import numpy as np
import math
from .misc import OPENCV_OBJECT_TRACKERS
from collections import deque
from util import DrawUtilities
from util import ColorUtilities


class TrackedObject:
   def __init__(self, type, buffer_size = 52, c=(255,0,0)):
      self.tracker = OPENCV_OBJECT_TRACKERS[type]()
      self.path = deque(maxlen=buffer_size)
      self.path_size = buffer_size
      self.bounding_box = None
      self.center = None
      self.fw = None
      self.fh = None
      self.enabled = True
      self.color = c
      
   def update(self,frame):
      if self.enabled:
         (success, box) = self.tracker.update(frame)
         self.bounding_box = box         
         if success:
            self.__track()
            self.__draw(frame)
            if not self.is_inside():
               self.enabled = False            

   def is_inside(self):
      (x, y, w, h) = [int(v) for v in self.bounding_box]
      left = x
      right = x + w
      top = y
      bottom = y + h
      #return (left >= 0 and right <= self.fw) and \
      #       (top >=0 and bottom <= self.fh)
      return (left >= 0 or left <= self.fw) and \
             (top >=0 and top <= self.fh)


   def __track(self):
      (x, y, w, h) = [int(v) for v in self.bounding_box]
      self.center = (math.floor(x+w/2),math.floor(y+h/2))   
      self.path.appendleft(self.center) 

   def __draw(self, frame):
      #get bbox center
      #track the point
      (x, y, w, h) = [int(v) for v in self.bounding_box]      
      #roi = frame[y:y+h,x:x+w].copy()   
      #DrawUtilities.drawRect(frame,x,y,w,h,10)
      DrawUtilities.drawCenter(frame,x,y,w,h,c=self.color)
      DrawUtilities.drawGrid(frame,x,y,w,h,c=self.color)
      DrawUtilities.drawPath(self.path, frame, n= self.path_size,c=self.color)
      #frame[y:y+h,x:x+w] = roi

   def init(self, box,frame):
      self.tracker.init(frame,box)
      (self.fh,self.fw,_) = frame.shape
      
   