# ================================================================================
# ================================================================================

# TODO:
# 1. module design
# 2. RGB-D fusion
# 3. normal estimation
# 4. franka integration

import time

import pyrealsense2 as rs
import numpy as np
import cv2

def get_connected_devices():
    ctx = rs.context()
    devices = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    if len(devices) < 2:
        raise RuntimeError("At least two RealSense devices are required!")
    return devices

def configure_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def main():

    serial_numbers = get_connected_devices()
    pipeline1 = configure_camera(serial_numbers[0])
    pipeline2 = configure_camera(serial_numbers[1])

    try:
        while True:
            start_time = time.perf_counter()

            frames1 = pipeline1.wait_for_frames()
            depth_frame1 = frames1.get_depth_frame()
            color_frame1 = frames1.get_color_frame()

            frames2 = pipeline2.wait_for_frames()
            depth_frame2 = frames2.get_depth_frame()
            color_frame2 = frames2.get_color_frame()

            if not depth_frame1 or not color_frame1 or not depth_frame2 or not color_frame2:
                continue

            # Convert frames to numpy arrays
            depth_image1 = np.asanyarray(depth_frame1.get_data())
            color_image1 = np.asanyarray(color_frame1.get_data())

            depth_image2 = np.asanyarray(depth_frame2.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())

            # Apply colormap to depth images for visualization
            depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image2, alpha=0.03), cv2.COLORMAP_JET)

            # Stack images for side-by-side display
            images1 = np.hstack((color_image1, depth_colormap1))
            images2 = np.hstack((color_image2, depth_colormap2))

            cv2.imshow('Camera 1', images1)
            cv2.imshow('Camera 2', images2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f'time elapsed: {elapsed_time} sec')
            
    finally:
        pipeline1.stop()
        pipeline2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()