# ================================================================================
# file name: background_filter.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import timer

class BackgroundFilter():
    """
    Background masking in RGBD data
    """
    # TODO: handle one cluster situation

    HEIGHT_ORIG = 480
    WIDTH_ORIG = 640
    DOWNSAMPLE_RATE = 0.40
    N_CLUSTERS = 2
    FOREGROUND_COLOR = [0, 255, 0]
    KMEANS = KMeans(n_clusters=N_CLUSTERS, 
                    max_iter=10, 
                    n_init=1,
                    algorithm='lloyd',
                    random_state=None)    # 42

    def __init__(self, mode):
        self.filters = {
            'fixedRange': self._fixed_range,
            'cdcluster': self._rgbd_clustering,
            'dcluster': self._depth_clustering,
            'ccluster': self._color_clustering,
            'dthresh': self._depth_thresholding,
            'cthresh': self._color_thresholding,
            'cdthresh': self._rgbd_thresholding
        }

        if mode not in self.filters:
            raise ValueError(f'Invalid mode: {mode}. Choose from {list(self.filters.keys())}')

        self.filter = self.filters[mode]
        
        self.color_frame = None
        self.depth_frame = None
        self.color_threshold = 50
        self.depth_threshold = 100


    def update_frames(self, color_frame:np.ndarray, depth_frame:np.ndarray):
        '''
        receive latest frame & downsample
        '''
        self.HEIGHT_ORIG = depth_frame.shape[0]
        self.WIDTH_ORIG = depth_frame.shape[1]
        height = round((1-self.DOWNSAMPLE_RATE) * self.HEIGHT_ORIG)
        width = round((1-self.DOWNSAMPLE_RATE) * self.WIDTH_ORIG)
        # self.color_frame = color_frame.copy()
        # self.depth_frame = depth_frame.copy()
        self.color_frame = cv2.resize(color_frame, (width, height))
        self.depth_frame = cv2.resize(depth_frame, (width, height))

    def _fixed_range(self):
        fixed_foreground_msk = np.zeros(self.depth_frame.shape, dtype=np.uint8)
        roi_h = round(self.depth_frame.shape[0]*0.5)   # 0.35
        roi_w = round(self.depth_frame.shape[1]*0.5)   # 0.35
        vert_offset = 35
        hori_offset = 18

        fixed_foreground_msk[round(self.depth_frame.shape[0]/2 + vert_offset - roi_h//2):
                             round(self.depth_frame.shape[0]/2 + vert_offset + roi_h//2),
                             round(self.depth_frame.shape[1]/2 + hori_offset - roi_w//2):
                             round(self.depth_frame.shape[1]/2 + hori_offset + roi_w//2)] = 1

        color_encoding = np.zeros_like(self.color_frame)
        color_encoding[fixed_foreground_msk == 1] = self.FOREGROUND_COLOR
        color_encoding = cv2.addWeighted(color_encoding,0.4,self.color_frame,0.5,0)

        return fixed_foreground_msk, color_encoding

    def _depth_thresholding(self):
        # simplified method: mask fixed area

        depth_foreground_msk = np.zeros(self.depth_frame.shape, dtype=np.uint8)
        roi_h = round(self.depth_frame.shape[0]*0.8)    # 0.8
        roi_w = round(self.depth_frame.shape[1]*0.8)    # 0.8

        seed_msk = np.zeros((roi_h+2, roi_w+2), dtype=np.uint8)
        seed_pnt = (roi_w//2, roi_h//2)

        # print(f'central depth: {self.depth_frame[roi_h//2, roi_w//2]}')

        # cv2.floodFill(self.color_frame[:roi_h, :roi_w], seed_msk, seed_pnt, 255,
        #              (self.color_threshold,)*3, (self.color_threshold,)*3,
        #              cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
        
        # print(f'color type: {self.color_frame.dtype}')
        # print(f'depth type: {self.depth_frame.dtype}')

        cv2.floodFill(self.depth_frame[:roi_h, :roi_w].astype(np.uint8), seed_msk, seed_pnt, 255,
                      loDiff=(10,),
                      upDiff=(20,),
                      flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

        depth_foreground_msk[round(self.depth_frame.shape[0]/2 - roi_h//2):
                             round(self.depth_frame.shape[0]/2 + roi_h//2),
                             round(self.depth_frame.shape[1]/2 - roi_w//2):
                             round(self.depth_frame.shape[1]/2 + roi_w//2)] = \
                                seed_msk[1:roi_h+1, 1:roi_w]

        color_encoding = np.zeros_like(self.color_frame)
        color_encoding[depth_foreground_msk == 1] = self.FOREGROUND_COLOR
        color_encoding = cv2.addWeighted(color_encoding,0.4,self.color_frame,0.5,0)

        return depth_foreground_msk, color_encoding
        

    def _color_thresholding(self):
        ...

    def _rgbd_thresholding(self):
        ...

    def _depth_clustering(self):
        depth_flat = self.depth_frame.flatten().reshape(-1, 1)

        dlabels = self.KMEANS.fit_predict(depth_flat)

        # determine foreground label by depth
        dlabels0 = dlabels == 0
        dlabels1 = dlabels == 1
        depth_foreground_msk = dlabels0 if np.mean(depth_flat[dlabels0]) < np.mean(depth_flat[dlabels1]) else dlabels1
        depth_foreground_msk = np.reshape(depth_foreground_msk, self.depth_frame.shape)

        color_encoding = np.zeros_like(self.color_frame)
        color_encoding[depth_foreground_msk == 1] = self.FOREGROUND_COLOR

        return depth_foreground_msk, color_encoding

    def _color_clustering(self, SMALL_FOREGROUND = True):
        rgb_flat = self.color_frame.reshape(-1, 3)

        clabels = self.KMEANS.fit_predict(rgb_flat)

        # determine foreground label by area
        clabels = np.reshape(clabels, self.depth_frame.shape)
        num_clabels0 = np.sum(clabels == 0)
        num_clabels1 = np.sum(clabels == 1)
        if SMALL_FOREGROUND:        # adjust based on target size
            clabelsFG = 0 if num_clabels0 < num_clabels1 else 1
        else: 
            clabelsFG = 0 if num_clabels0 > num_clabels1 else 1
        color_foreground_msk = (clabels==clabelsFG)

        color_encoding = np.zeros_like(self.color_frame)
        color_encoding[color_foreground_msk == 1] = self.FOREGROUND_COLOR
        color_encoding = cv2.addWeighted(color_encoding,0.4,self.color_frame,0.5,0)

        return color_foreground_msk, color_encoding

    def _rgbd_clustering(self):
        rgb_flat = self.color_frame.reshape(-1, 3)
        depth_flat = self.depth_frame.flatten().reshape(-1, 1)
        clabels = self.KMEANS.fit_predict(rgb_flat)
        dlabels = self.KMEANS.fit_predict(depth_flat)

        # determine foreground label in depth frame
        dlabels0 = dlabels == 0
        dlabels1 = dlabels == 1
        depth_foreground_msk = dlabels0 if np.mean(depth_flat[dlabels0]) < np.mean(depth_flat[dlabels1]) else dlabels1
        depth_foreground_msk = np.reshape(depth_foreground_msk, self.depth_frame.shape)

        # determine foreground label in color frame
        clabels = np.reshape(clabels, self.depth_frame.shape)
        num_clabels0 = np.sum(clabels[depth_foreground_msk] == 0)
        num_clabels1 = np.sum(clabels[depth_foreground_msk] == 1)
        clabelsFG = 0 if num_clabels0 > num_clabels1 else 1
        color_foreground_msk = clabels==clabelsFG

        # combine depth & color mask
        rgbd_foreground_msk = np.bitwise_and(depth_foreground_msk, color_foreground_msk)
        color_encoding = np.zeros_like(self.color_frame)
        color_encoding[rgbd_foreground_msk == 1] = self.FOREGROUND_COLOR

        return rgbd_foreground_msk, color_encoding

    # @ timer
    def process(self, color_frame, depth_frame):
        self.update_frames(color_frame, depth_frame)
        foreground_msk, color_encoding = self.filter()
        foreground_msk_orig = cv2.resize(foreground_msk.astype(np.uint8), 
                                         (self.WIDTH_ORIG, self.HEIGHT_ORIG), 
                                         interpolation=cv2.INTER_CUBIC)
        color_encoding_orig = cv2.resize(color_encoding.astype(np.uint8), 
                                         (self.WIDTH_ORIG, self.HEIGHT_ORIG), 
                                         interpolation=cv2.INTER_CUBIC)
        # TODO: add post-processing to remove noise
        return foreground_msk_orig, color_encoding_orig

