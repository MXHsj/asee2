import os

import numpy as np
import open3d as o3d

from fit_surface import FitQuadraticSurface
from utils import filter_pcd_outliers

def create_grid_lines(points, rows, cols):
    """
    Create grid lines for a structured grid of points.

    Args:
        points (numpy.ndarray): Array of points (shape: Nx3).
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        line_set (open3d.geometry.LineSet): LineSet object representing the grid.
    """
    # Generate line connections (horizontal and vertical)
    lines = []
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j  # Current point index
            # Horizontal connection (to the right neighbor)
            if j < cols - 1:
                lines.append([idx, idx + 1])
            # Vertical connection (to the bottom neighbor)
            if i < rows - 1:
                lines.append([idx, idx + cols])

    colors = [[0, 0, 1] for _ in lines]  # Blue lines

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

cam1_pcd_name = 'example_cam1_pcd'
cam2_pcd_name = 'example_cam2_pcd'
# cam1_pcd_name = 'cam1_pcd'
# cam2_pcd_name = 'cam2_pcd'
# cam1_pcd_name = 'merged_pcd'
            
cam1_pcd_path = os.path.dirname(__file__) + '/' + cam1_pcd_name + '.npy'
cam2_pcd_path = os.path.dirname(__file__) + '/' + cam2_pcd_name + '.npy'

cam1_pcd_raw = np.load(cam1_pcd_path)
cam2_pcd_raw_ = np.load(cam2_pcd_path)

surf_fitter = FitQuadraticSurface()

# ===== transformations =====
T_CAM1_CAM2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0.12544],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

T_F_PROBE = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.224],
                      [0, 0, 0, 1]])

# transformation from cam1 to probe tip
T_CAM1_PROBE = np.array([[-0.9997596514674911, 0.01895868516356237, -0.011009430251793509, 0.003494285383665675],
                         [-0.019020413904005103, -0.9998038049115955, 0.005529515278539474, 0.04646526225586623],
                         [-0.010902437916377784, 0.005737590187891416, 0.9999241055731759, 0.17898964143239388],
                         [0, 0, 0, 1]])

cam2_pcd_raw = cam2_pcd_raw_ + T_CAM1_CAM2[0:3, -1].T
P_CAM1 = T_CAM1_PROBE[:3, -1]   # probe tip pose w.r.t cam1
print('probe tip position', P_CAM1)
# ===========================

# ===== filter pcd by z-axis =====
cam1_pcd_raw = filter_pcd_outliers(cam1_pcd_raw)
cam2_pcd_raw = filter_pcd_outliers(cam2_pcd_raw)
# ================================

# ===== for visualization, flip z-axis =====
cam1_pcd_raw[:,-1] = -cam1_pcd_raw[:,-1]
cam2_pcd_raw[:,-1] = -cam2_pcd_raw[:,-1]
# ==========================================

# ===== fit tangent plane =====
combined_pcd = np.vstack([cam1_pcd_raw, cam2_pcd_raw])
xrange = [np.min(combined_pcd[:,0]), np.max(combined_pcd[:,0])]
yrange = [np.min(combined_pcd[:,1]), np.max(combined_pcd[:,1])]
coeffs = surf_fitter.fit_quadratic_plane(combined_pcd)
norm = surf_fitter.calculate_quadratic_surface_normal(coeffs, x=P_CAM1[0], y=P_CAM1[1])

qs_pcd_raw = surf_fitter.sample_quadratic_surface(coeffs, xrange, yrange, resolution=30)
# print('quadratic plane coeffs', coeffs)
print('normal vector: ', norm)
# =============================

# ===== draw cam1 & cam2 pcd =====
cam1_pcd = o3d.geometry.PointCloud()
cam1_pcd.points = o3d.utility.Vector3dVector(cam1_pcd_raw)

cam2_pcd = o3d.geometry.PointCloud()
cam2_pcd.points = o3d.utility.Vector3dVector(cam2_pcd_raw)
# ================================

# ===== draw fitted quadratic surface =====
qs_pcd = o3d.geometry.PointCloud()
qs_pcd.points = o3d.utility.Vector3dVector(qs_pcd_raw)
qs_color = np.zeros_like(qs_pcd_raw)
qs_pcd.colors = o3d.utility.Vector3dVector(qs_color)
qs_grid = create_grid_lines(qs_pcd_raw, rows=30, cols=30)
# =========================================

# ===== draw normal vector =====
mag = -0.04
vec_start = P_CAM1
vec_start[2] = -vec_start[2]
vec_end = vec_start + mag * norm
norm_vec = o3d.geometry.LineSet()
norm_vec.points = o3d.utility.Vector3dVector([vec_start, vec_end])
norm_vec.lines = o3d.utility.Vector2iVector([[0, 1]])
norm_vec.colors = o3d.utility.Vector3dVector([[1,0,0]])
# ==============================

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cam1_pcd)
vis.add_geometry(cam2_pcd)
vis.add_geometry(qs_pcd)
vis.add_geometry(qs_grid)
vis.add_geometry(norm_vec)

opt = vis.get_render_option()
opt.show_coordinate_frame = True
opt.point_size = 5
opt.line_width = 20

vis.run()
vis.destroy_window()
