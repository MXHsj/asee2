# ================================================================================
# file name: asee2_rosbridge.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================

# import rospy
import cv2

from asee2 import ASEE2
from utils import timer

ee = ASEE2()

try:
    while True:
        ee.onUpdate()

        cv2.imshow('color frame', ee.visualize_color_frames())
        cv2.imshow('depth frame', ee.visualize_depth_frames())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    ee.onFinish()

