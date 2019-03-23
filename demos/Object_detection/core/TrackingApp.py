from deco import exception
import uuid
from util import ColorUtilities
from util import DrawUtilities
from cv2 import Tracker
from .TrackedObject import TrackedObject
from enum import Enum, auto
from collections import deque
from imutils.video import FPS, VideoStream
import math
import time
import cv2
import sys
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir)

# print(inspect.getmembers(Tracker))


class TrackingApp():

    def __init__(self, src, tracker="kcf", output_video_path="out.avi"):
        self.tracking_objects = []
        self.tracking_method = tracker
        self.is_recording = False
        self.focus_mode = False
        self.blurring_kernel = (25, 25)
        self.window_name = str(uuid.uuid4())
        self.frame = None
        self.video_capture_type = None
        if isinstance(src, int):
            self.video_capture = cv2.VideoCapture(src)
            self.fps = 1
            self.video_capture_type = 1
        elif isinstance(src, str) and os.path.isfile(src):
            self.video_capture = cv2.VideoCapture(src)
            self.fps = 25
            self.video_capture_type = 2
        else:
            raise Exception("Invalid video source")
        self.frame_width = int(self.video_capture.get(3))
        self.frame_height = int(self.video_capture.get(4))
        print(self.frame_width, self.frame_height)
        # self.video_recorder = cv2.VideoWriter(
        # output_video_path,
        # cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))

    def select_object(self):
        bounding_box = cv2.selectROI(
            self.window_name,
            self.frame,
            fromCenter=False,
            showCrosshair=True)
        return bounding_box

    def process_frame(self):
        if self.video_capture_type == 1:
            self.frame = cv2.flip(self.frame, 1)
        #self.frame = cv2.GaussianBlur(
            #self.frame, self.blurring_kernel, 0)

    def update_frame(self):
        for i, tracker in enumerate(self.tracking_objects):
            tracker.update(self.frame)

    def display_frame(self):
        cv2.imshow(self.window_name, self.frame)

    def handlerKeyEvents(self, key):
        if key == ord("q"):
            self.stop()
        elif key == ord("1"):
            try:
                bounding_box = self.select_object()
                method = self.tracking_method
                color = ColorUtilities.rand_color()
                tracked_object = TrackedObject(method, c=color)
                tracked_object.init(bounding_box, self.frame)
                self.tracking_objects.append(tracked_object)
            except:
                pass
        elif key == ord("2"):
            self.tracking_objects.clear()

    def start(self):
        try:
            if (self.video_capture.isOpened() == False):
                raise Exception("Error opening video stream or file")
            while(True):
                _, self.frame = self.video_capture.read()
                if self.frame is None:
                    break
                self.process_frame()
                self.update_frame()
                self.display_frame()
                # self.video_recorder.write(self.frame)
                self.handlerKeyEvents(cv2.waitKey(self.fps) % 255)
        except Exception as ex:
            print("Error gathering the frame {}".format(ex))
            self.stop()

    def stop(self):
        self.video_capture.release()
        # self.video_recorder.release()
        cv2.destroyAllWindows()
