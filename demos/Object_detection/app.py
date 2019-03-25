import cv2
import sys
import time
import math
from imutils.video import FPS, VideoStream
from collections import deque
from core import TrackingApp
from multiprocessing import Pool, Process
import subprocess

video_file = r"D:\TexasAandM\study\David\AI-Study-PoCs\demos\Object_detection\videos\traffic3.mp4"
#TrackingApp(0).start()
TrackingApp(video_file).start()


