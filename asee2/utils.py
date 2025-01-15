import time

import numpy as np

def timer(func): 
    def wrap_func(*args, **kwargs): 
        t1 = time.perf_counter()
        ret = func(*args, **kwargs) 
        t2 = time.perf_counter()
        print(f'function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return ret 
    return wrap_func

def se3_inv(mat_in: np.ndarray):
    R = mat_in[:3, :3]
    t = mat_in[:3, -1]
    mat_out = np.zeros_like(mat_in)
    mat_out[:3, :3] = R.T
    mat_out[:3, -1] = -R.T @ t
    mat_out[-1, -1] = 1
    return mat_out

def filter_pcd_outliers(pcd_in):
    x_lower_msk = pcd_in[:,0] > np.mean(pcd_in[:,0]) - 1.5*np.std(pcd_in[:,0])
    x_upper_msk = pcd_in[:,0] < np.mean(pcd_in[:,0]) + 1.5*np.std(pcd_in[:,0])
    y_lower_msk = pcd_in[:,1] > np.mean(pcd_in[:,1]) - 1.5*np.std(pcd_in[:,1])
    y_upper_msk = pcd_in[:,1] < np.mean(pcd_in[:,1]) + 1.5*np.std(pcd_in[:,1])
    z_lower_msk = pcd_in[:,2] > np.mean(pcd_in[:,2]) - 1.0*np.std(pcd_in[:,2])
    z_upper_msk = pcd_in[:,2] < np.mean(pcd_in[:,2]) + 1.0*np.std(pcd_in[:,2])
    x_msk = np.bitwise_and(x_lower_msk, x_upper_msk)
    y_msk = np.bitwise_and(y_lower_msk, y_upper_msk)
    z_msk = np.bitwise_and(z_lower_msk, z_upper_msk)
    xyz_msk = np.bitwise_and(np.bitwise_and(x_msk, y_msk), z_msk)
    pcd_out = pcd_in[xyz_msk]
    return pcd_out