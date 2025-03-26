# ================================================================================
# file name: asee2.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================

import os

import pyrealsense2 as rs
import numpy as np
import cv2

from asee2_core.background_filter import BackgroundFilter
from asee2_core.fit_surface import FitQuadraticSurface
from asee2_core.utils import timer, filter_pcd_outliers
from asee2_core.constants import CAM1_T_CAM2, CAM1_T_PROBE, devices

def data_cursor(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f'x: {x}')
        print(f'y: {y}')

class ASEE2():
    
    P_CAM1 = CAM1_T_PROBE[:3, -1]   # probe tip pose w.r.t cam1

    FRM_HEIGHT = 480
    FRM_WIDTH = 640
    MIN_DIST = 0.07     # [m]
    MAX_DIST = 0.25     # 0.50 [m]
    PCD_DOWNSAMPLE_FACTOR = 2e-4
    MIN_NUM_PNTS = 300
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

    cam1_pcd = None
    cam2_pcd = None
    surf_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # quadratic surface fitting
    normal_vector = np.array([0.0, 0.0, 1.0])

    _pcd_buffer = np.zeros((FRM_HEIGHT*FRM_WIDTH, 3), np.float32)

    bg_filter = BackgroundFilter('fixedRange')
    # bg_filter = BackgroundFilter('dthresh')
    # bg_filter = BackgroundFilter('ccluster')
    surf_fitter = FitQuadraticSurface()
    dthresh_filter = rs.threshold_filter(min_dist=0.07, max_dist=0.50)

    isMskBg = True

    def __init__(self):
        self.get_connected_devices()
        self.camera1, self.cam1_intri = self._configure_camera(devices[0])
        self.camera2, self.cam2_intri = self._configure_camera(devices[1])
        
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
    def convert_rgbd_to_pcd(self, tissue_msk: np.ndarray, depth_raw, intri):
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
        # TODO: handle empty pnts
        # print(f'number of points: {pnts.shape[0]}')
        self._pcd_buffer[:] = 0
        return pnts

    def process_pcd(self, pcd_in):
        pcd_out = filter_pcd_outliers(pcd_in)
        return pcd_out

    def merge_pcds(self, cam1_pcd:np.ndarray, cam2_pcd:np.ndarray) -> np.ndarray:
        cam2_pcd_transformed = cam2_pcd + CAM1_T_CAM2[:3, -1].T
        merged_pcd = np.vstack([cam1_pcd, cam2_pcd_transformed])
        return merged_pcd

    def get_connected_devices(self):
        ctx = rs.context()
        devices = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        if len(devices) < 2:
            raise RuntimeError("At least two RealSense devices are required!")
        # TODO: allow updating devices

    def get_cam1_color(self) -> np.ndarray:
        return self.cam1_color
    
    def get_cam2_color(self) -> np.ndarray:
        return self.cam2_color
    
    def get_cam1_depth(self) -> np.ndarray:
        return self.cam1_depth
    
    def get_cam2_depth(self) -> np.ndarray:
        return self.cam2_depth

    def get_normal_vector(self) -> np.ndarray:
        return self.normal_vector
    
    def get_surf_coeffs(self) -> np.ndarray:
        return self.surf_coeffs
    
    def get_merged_pcd(self) -> np.ndarray:
        return self.merged_pcd

    def visualize_color_frames(self):
        color_frame = np.vstack((self.cam1_color, self.cam2_color))
        return color_frame

    def visualize_depth_frames(self):
        depth_frame = np.vstack((self.cam1_depth, self.cam2_depth))
        color_coded = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        return color_coded

    def onUpdate(self):
        cam1_color, cam1_depth, cam1_depth_raw = self._stream_camera(self.camera1)
        cam2_color, cam2_depth, cam2_depth_raw = self._stream_camera(self.camera2)
        if cam1_color is not None and cam1_depth is not None and cam2_color is not None and cam2_depth is not None:
            self.cam1_color = cam1_color
            self.cam1_depth = cam1_depth
            self.cam2_color = cam2_color
            self.cam2_depth = cam2_depth

        if self.isMskBg:
            tissue_msk1, tissue_msk1_colorized = self.bg_filter.process(self.cam1_color, self.cam1_depth)
            tissue_msk2, tissue_msk2_colorized = self.bg_filter.process(np.flipud(self.cam2_color), np.flipud(self.cam2_depth))
            tissue_msk2 = np.flipud(tissue_msk2)
            tissue_msk2_colorized = np.flipud(tissue_msk2_colorized)

        else:
            tissue_msk1 = np.ones_like(cam1_depth, np.uint8)
            tissue_msk1_colorized = np.ones_like(cam1_depth, np.uint8)
            tissue_msk2 = np.ones_like(cam2_depth, np.uint8)
            tissue_msk2_colorized = np.ones_like(cam1_depth, np.uint8)

        cam1_pcd_raw = self.convert_rgbd_to_pcd(tissue_msk1,
                                                cam1_depth_raw, 
                                                self.cam1_intri)
        
        cam2_pcd_raw = self.convert_rgbd_to_pcd(tissue_msk2,
                                                cam2_depth_raw, 
                                                self.cam2_intri)
        
        self.cam1_pcd = self.process_pcd(cam1_pcd_raw)
        self.cam2_pcd = self.process_pcd(cam2_pcd_raw)
        self.merged_pcd = self.merge_pcds(self.cam1_pcd, self.cam2_pcd)
        self.merged_pcd = self.process_pcd(self.merged_pcd)

        num_pnts = self.merged_pcd.shape[0]
        if num_pnts < self.MIN_NUM_PNTS:
            # print(f'no target')
            self.surf_coeffs = np.zeros((6,))
            self.normal_vector = np.array([0.0, 0.0, 0.0])
            
        else:
            self.surf_coeffs = self.surf_fitter.fit_quadratic_plane(self.merged_pcd)
            self.normal_vector = self.surf_fitter.calculate_quadratic_surface_normal(self.surf_coeffs, 
                                                                                x=self.P_CAM1[0], 
                                                                                y=self.P_CAM1[1])
        
        # print(f'normal vector: {self.normal_vector}')

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

        # ========== test bg filtering ==========
        tissue_msk_colorized = np.vstack((tissue_msk1_colorized, tissue_msk2_colorized))
        dbg_img = tissue_msk_colorized.copy()
        # =======================================

        return dbg_img

    def onFinish(self):
        self.camera1.stop()
        self.camera2.stop()
        # self._dump_pcd(self.cam1_pcd, 'cam1_pcd')
        # self._dump_pcd(self.cam2_pcd, 'cam2_pcd')


if __name__ == "__main__":
    asee2 = ASEE2()
    # cv2.namedWindow('color frame')
    # cv2.setMouseCallback('color frame', data_cursor)
    try:
        while True:
            dbg_img = asee2.onUpdate()

            cv2.imshow('bg filtered', dbg_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2.imshow('color frame', self.visualize_color_frames())
            # cv2.imshow('depth frame', self.visualize_depth_frames())
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        asee2.onFinish()