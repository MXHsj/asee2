# ================================================================================
# file name: asee2.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================

import os
import threading

import pyrealsense2 as rs
import numpy as np
import cv2

from background_filter import BackgroundFilter
from fit_surface import FitQuadraticSurface
from utils import timer, filter_pcd_outliers


def data_cursor(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f'x: {x}')
        print(f'y: {y}')

class ASEE2():
    '''
    TODO:
    - test in real-time
    - robotic integration
    '''
    T_F_PROBE = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.224],
                          [0, 0, 0, 1]])
    
    # transformation from cam1 to cam2
    T_CAM1_CAM2 = np.array([[1, 0, 0, 0], #[[1, 0, 0, 0.165+0.018],
                            [0, 1, 0, 0.12544],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    # transformation from cam1 to probe tip
    T_CAM1_PROBE = np.array([[-0.9997596514674911, 0.01895868516356237, -0.011009430251793509, 0.003494285383665675],
                             [-0.019020413904005103, -0.9998038049115955, 0.005529515278539474, 0.04646526225586623],
                             [-0.010902437916377784, 0.005737590187891416, 0.9999241055731759, 0.17898964143239388],
                             [0, 0, 0, 1]])
    
    P_CAM1 = T_CAM1_PROBE[:3, -1]   # probe tip pose w.r.t cam1

    FRM_HEIGHT = 480
    FRM_WIDTH = 640
    MIN_DIST = 0.07     # [m]
    MAX_DIST = 0.50     # [m]
    PCD_DOWNSAMPLE_FACTOR = 2e-4
    devices = ['130322273859', '128422271677']            # device serial number
    camera1 = None
    camera2 = None
    cam1_intri = None
    cam2_intri = None
    # cam1_depth_raw = None
    cam1_depth = np.zeros((FRM_HEIGHT, FRM_WIDTH), np.uint16)
    cam1_color = np.zeros((FRM_HEIGHT, FRM_WIDTH), np.uint8)
    # cam2_depth_raw = None
    cam2_depth = np.zeros((FRM_HEIGHT, FRM_WIDTH), np.uint16)
    cam2_color = np.zeros((FRM_HEIGHT, FRM_WIDTH), np.uint8)

    _pcd_buffer = np.zeros((FRM_HEIGHT*FRM_WIDTH, 3), np.float32)

    bg_filter = BackgroundFilter('ccluster')
    surf_fitter = FitQuadraticSurface()
    dthresh_filter = rs.threshold_filter(min_dist=0.07, max_dist=0.50)

    def __init__(self):
        self.get_connected_devices()
        self.camera1, self.cam1_intri = self._configure_camera(self.devices[0])
        self.camera2, self.cam2_intri = self._configure_camera(self.devices[1])
        
    def _configure_camera(self, serial_number):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, self.FRM_WIDTH, self.FRM_HEIGHT, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.FRM_WIDTH, self.FRM_HEIGHT, rs.format.bgr8, 30)
        cfg = pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)                       # Fetch stream profile for depth stream
        intrinsics = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        return pipeline, intrinsics
    
    def _stream_camera(self, camera):
        frame = camera.wait_for_frames()
        depth_frame_raw = frame.get_depth_frame()
        color_frame_raw = frame.get_color_frame()

        depth_frame = np.asanyarray(depth_frame_raw.get_data())
        color_frame = np.asanyarray(color_frame_raw.get_data())

        return color_frame, depth_frame, depth_frame_raw

    def _dump_pcd(self, pcd: np.ndarray, pcd_name):
        pcd_path = os.path.dirname(__file__) + '/' + pcd_name + '.npy'
        with open(pcd_path, 'wb') as f:
            np.save(f, pcd)
            print('pcd saved to: ', pcd_path)

    # @timer
    def convert_rgbd_to_pcd(self, color_frame: np.ndarray, depth_frame: np.ndarray, depth_raw, intri, isMskBg=True):
        if isMskBg:
            tissue_msk, _ = self.bg_filter.process(color_frame, depth_frame)
        else:
            tissue_msk = np.ones_like(depth_frame, np.uint8)
        [row, col] = np.where(tissue_msk == 1)
        num_pix = row.shape[0]
        sample_interv = round(np.max([1, self.PCD_DOWNSAMPLE_FACTOR * num_pix]))

        for n in range(0, num_pix, sample_interv):
            depth = depth_raw.get_distance(col[n],row[n])
            if depth > self.MIN_DIST and depth < self.MAX_DIST:
                point = rs.rs2_deproject_pixel_to_point(intri, 
                                                        [col[n], row[n]], 
                                                        depth)
                self._pcd_buffer[n, :] = point
        
        pnts = self._pcd_buffer[~np.all(self._pcd_buffer == 0, axis=1)]
        self._pcd_buffer[:] = 0

        return pnts

    def process_pcd(self, pcd_in):
        pcd_out = filter_pcd_outliers(pcd_in)
        return pcd_out

    def merge_pcds(self, cam1_pcd:np.ndarray, cam2_pcd:np.ndarray):
        cam2_pcd_transformed = cam2_pcd + self.T_CAM1_CAM2[0:3, -1].T
        merged_pcd = np.vstack([cam1_pcd, cam2_pcd_transformed])
        return merged_pcd

    def get_connected_devices(self):
        ctx = rs.context()
        devices = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        if len(devices) < 2:
            raise RuntimeError("At least two RealSense devices are required!")
        # TODO: allow updating devices

    def visualize_color_frames(self):
        color_frame = np.vstack((self.cam1_color, self.cam2_color))
        return color_frame

    def visualize_depth_frames(self):
        depth_frame = np.vstack((self.cam1_depth, self.cam2_depth))
        color_coded = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        return color_coded

    def onUpdate(self):
        try:
            # cv2.namedWindow('color frame')
            # cv2.setMouseCallback('color frame', data_cursor)
            while True:
                cam1_color, cam1_depth, cam1_depth_raw = self._stream_camera(self.camera1)
                cam2_color, cam2_depth, cam2_depth_raw = self._stream_camera(self.camera2)
                if cam1_color is None or cam1_depth is None or cam2_color is None or cam2_depth is None:
                    continue
                else:
                    self.cam1_color = cam1_color
                    self.cam1_depth = cam1_depth
                    self.cam2_color = cam2_color
                    self.cam2_depth = cam2_depth
                
                # ========== test bg filtering ==========
                # _, tissue_msk1_colorized = self.bg_filter.process(self.cam1_color, self.cam1_depth)
                # _, tissue_msk2_colorized = self.bg_filter.process(self.cam2_color, self.cam2_depth)
                # tissue_msk_colorized = np.vstack((tissue_msk1_colorized, tissue_msk2_colorized))
                # cv2.imshow('bg filtered', tissue_msk_colorized)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # =======================================

                # ========== test rgbd to pcd ==========
                cam1_pcd_raw = self.convert_rgbd_to_pcd(cam1_color, cam1_depth,
                                                        cam1_depth_raw, 
                                                        self.cam1_intri)
                
                cam2_pcd_raw = self.convert_rgbd_to_pcd(cam2_color, cam2_depth,
                                                        cam2_depth_raw, 
                                                        self.cam2_intri)
                
                cam1_pcd = self.process_pcd(cam1_pcd_raw)
                cam2_pcd = self.process_pcd(cam2_pcd_raw)
                merged_pcd = self.merge_pcds(cam1_pcd, cam2_pcd)
                coeffs = self.surf_fitter.fit_quadratic_plane(merged_pcd)
                norm = self.surf_fitter.calculate_quadratic_surface_normal(coeffs, 
                                                                           x=self.P_CAM1[0], 
                                                                           y=self.P_CAM1[1])
                print(f'normal vector: {norm}')
                # ======================================

                # ===== calibrate probe tip pos =====
                # probe_cam1 = rs.rs2_deproject_pixel_to_point(self.cam1_intri, 
                #                                             [347, 376], 
                #                                             cam1_depth_raw.get_distance(347, 376))
                # probe_cam2 = rs.rs2_deproject_pixel_to_point(self.cam2_intri, 
                #                                             [342, 587-480], 
                #                                             cam1_depth_raw.get_distance(342, 587-480))
                # print(f'probe tip pos in cam1: {probe_cam1}')
                # print(f'probe tip pos in cam2: {probe_cam2}')
                # ===================================

                # cv2.imshow('color frame', self.visualize_color_frames())
                # cv2.imshow('depth frame', self.visualize_depth_frames())
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        finally:
            self._dump_pcd(cam1_pcd, 'cam1_pcd')
            self._dump_pcd(cam2_pcd, 'cam2_pcd')
            # self._dump_pcd(merged_pcd, 'merged_pcd')
            self.onFinish()

    def onFinish(self):
        self.camera1.stop()
        self.camera2.stop()


if __name__ == "__main__":
    asee2 = ASEE2()
    asee2.onUpdate()