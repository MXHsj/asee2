# ================================================================================
# file name: asee2_rosbridge.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================
import numpy as np
import cv2

import rospy
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray

# ===== for rviz visualization =====
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
# ==================================

from asee2_core.asee2 import ASEE2
from asee2_core.utils import timer

class ASEE2ROSBridge():

  def __init__(self, rate=30, isVis=True):
    self.isVis = isVis
    self.ee = ASEE2()
    self.norm = np.array([0.0, 0.0, 1.0])
    self.surf_coeff = np.zeros((6,))
    self.merged_pcd = np.zeros((3,))
    
    rospy.init_node('asee2_rosbridge', anonymous=True)
    self.merged_pcd_pub = rospy.Publisher('/asee2/merged/pcd', PointCloud2, queue_size=100)

    self.cam1_rgb_pub = rospy.Publisher('/asee2/cam1/rgb', Image, queue_size=100)
    self.cam2_rgb_pub = rospy.Publisher('/asee2/cam2/rgb', Image, queue_size=100)
    self.cam1_depth_pub = rospy.Publisher('/asee2/cam1/depth', Image, queue_size=100)
    self.cam2_depth_pub = rospy.Publisher('/asee2/cam2/depth', Image, queue_size=100)

    self.surface_coeff_pub = rospy.Publisher('/asee2/surface/coeff', Float64MultiArray, queue_size=100)
    
    self.norm_pub = rospy.Publisher('/asee2/normal_vector', Vector3Stamped, queue_size=100)
    self.norm_vis_pub = rospy.Publisher('asee2/normal_vector_vis', Marker, queue_size=100)

    self.rate = rospy.Rate(rate)
  
  def generate_color_msg(self, data:np.ndarray, frame_id:str) -> Image:
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = "bgr8"        # use rgb8 for RGB data
    msg.is_bigendian = False
    msg.step = msg.width * 3 
    msg.data = data.tobytes()

    return msg
  
  def generate_depth_msg(self, data:np.ndarray, frame_id:str) -> Image:
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = "mono16"       # single channel uint16
    msg.is_bigendian = False
    msg.step = msg.width * 1 
    msg.data = data.tobytes()

    return msg
  
  def generate_pc2_msg(self, data:np.ndarray, frame_id:str) -> PointCloud2:
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    msg = pc2.create_cloud_xyz32(header, data)

    return msg
  
  def generate_normal_msg(self, data:np.ndarray, frame_id:str) -> Vector3Stamped:
    msg = Vector3Stamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.vector.x = self.norm[0]
    msg.vector.y = self.norm[1]
    msg.vector.z = self.norm[2]

    return msg
  
  def generate_normal_vis_msg(self, data:np.ndarray, frame_id:str) -> Marker:
    norm_marker = Marker()
    norm_marker.header.frame_id = frame_id
    norm_marker.header.stamp = rospy.Time.now()
    norm_marker.ns = "normal_vector"
    norm_marker.id = 0
    norm_marker.type = Marker.ARROW
    norm_marker.action = Marker.ADD
    norm_marker.scale.x = 0.01  # shaft diameter
    norm_marker.scale.y = 0.03   # head diameter
    norm_marker.scale.z = 0.0   # unused for arrows
    norm_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red

    start_point = Point(np.mean(self.merged_pcd[:,0]), 
                        np.mean(self.merged_pcd[:,1]), 
                        np.min(self.merged_pcd[:,2]))

    end_point = Point(data[0], 
                      data[1], 
                      data[2])

    norm_marker.points = [
      start_point,
      end_point
    ]

    return norm_marker

  def onUpdate(self):
    try:
      while not rospy.is_shutdown():
        self.ee.onUpdate()
        self.norm = self.ee.get_normal_vector()
        self.surf_coeff = self.ee.get_surf_coeffs()
        self.merged_pcd = self.ee.get_merged_pcd()

        self.cam1_color = self.ee.get_cam1_color()
        self.cam2_color = self.ee.get_cam2_color()
        self.cam1_depth = self.ee.get_cam1_depth()
        self.cam2_depth = self.ee.get_cam2_depth()

        # ===== publish color frames =====
        cam1_rgb_msg = self.generate_color_msg(data=self.cam1_color, frame_id='asee2')
        self.cam1_rgb_pub.publish(cam1_rgb_msg)

        cam2_rgb_msg = self.generate_color_msg(data=self.cam2_color, frame_id='asee2')
        self.cam2_rgb_pub.publish(cam2_rgb_msg)
        # ================================

        # ===== publish depth frames =====
        cam1_depth_msg = self.generate_depth_msg(data=self.cam1_depth, frame_id='asee2')
        self.cam1_depth_pub.publish(cam1_depth_msg)

        cam2_depth_msg = self.generate_depth_msg(data=self.cam2_depth, frame_id='asee2')
        self.cam2_depth_pub.publish(cam2_depth_msg)
        # ================================

        # ===== publish merged pointcloud =====
        merged_pcd_msg = self.generate_pc2_msg(data=self.merged_pcd, frame_id='asee2')
        self.merged_pcd_pub.publish(merged_pcd_msg)
        # =====================================

        # ===== publish normal vector & surface params =====
        norm_msg = self.generate_normal_msg(data=self.norm, frame_id='asee2')
        self.norm_pub.publish(norm_msg)

        norm_vis_msg = self.generate_normal_vis_msg(data=self.norm, frame_id='asee2')
        self.norm_vis_pub.publish(norm_vis_msg)

        self.surface_coeff_pub.publish(Float64MultiArray(data=self.surf_coeff))
        # ==================================================

        if self.isVis:
          vis = np.hstack((self.ee.visualize_color_frames(), self.ee.visualize_depth_frames()))
          cv2.imshow('color & depth frame', vis)
          # cv2.imshow('depth frame', self.ee.visualize_depth_frames())
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        self.rate.sleep()

    finally:
      self.ee.onFinish()
      print(f'ASEE2.0 ROS bridge shutdown')

if __name__ == "__main__":
    bridge = ASEE2ROSBridge(isVis=False)
    bridge.onUpdate()
    