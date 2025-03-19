# ================================================================================
# file name: asee2_rosbridge.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from asee2 import ASEE2
from utils import timer

class ASEE2ROSBridge():

  def __init__(self, rate=30):
    self.ee = ASEE2()
    self.norm = np.array([0.0, 0.0, 1.0])
    self.surf_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    rospy.init_node('asee2_rosbridge', anonymous=True)
    # self.cam1_pcd_pub = rospy.Publisher('/asee2/cam1/pcd', PointCloud2, queue_size=100)
    self.cam1_rgb_pub = rospy.Publisher('/asee2/cam1/rgb', Image, queue_size=100)
    # self.cam2_pcd_pub = rospy.Publisher('/asee2/cam2/pcd', PointCloud2, queue_size=100)
    self.cam2_rgb_pub = rospy.Publisher('/asee2/cam2/rgb', Image, queue_size=100)

    self.surface_coeff_pub = rospy.Publisher('/asee2/surface/coeff', Float64MultiArray, queue_size=100)
    
    self.combined_pcd_pub = rospy.Publisher('/asee2/pcd', PointCloud2, queue_size=100)
    self.norm_pub = rospy.Publisher('/asee2/normal_vector', Float64MultiArray, queue_size=1)

    self.rate = rospy.Rate(rate)
    
  def onUpdate(self):
    try:
      while not rospy.is_shutdown():
        self.ee.onUpdate()
        self.norm = self.ee.get_normal_vector()
        self.surf_coeff = self.ee.get_surf_coeffs()

        self.norm_pub.publish(Float64MultiArray(data=self.norm))
        self.surface_coeff_pub.publish(Float64MultiArray(data=self.surf_coeff))

        # cv2.imshow('color frame', self.ee.visualize_color_frames())
        cv2.imshow('depth frame', self.ee.visualize_depth_frames())
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        self.rate.sleep()

    finally:
      self.ee.onFinish()
      print(f'asee2 ROS bridge shutdown')

if __name__ == "__main__":
    bridge = ASEE2ROSBridge()
    bridge.onUpdate()
    