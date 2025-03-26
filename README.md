# Active-Sensing End-Effector (ASEE) 2.0
End effector for robotic ultrasound applications

See our first generation prototype [```ASEE1.0```](https://ieeexplore.ieee.org/abstract/document/9932673).

## Installation
### 0. dependencies

```ASEE2.0``` is developed and tested on Ubuntu 20.04 & ROS Noetic.

### 1. install python packages
```sh
pip3 install numpy
pip3 install scikit-learn
pip3 install pyrealsense2
pip3 install open3d
```

## Usage

### 0. launch the driver
```sh
roslaunch asee2 asee2_bringup.launch
```

### 1. robot-integration: achieving real-time autonomous orientation control for robotic ultrasound



### 1. ROS topics
| name | description |
| :---: | :---: |
| ```/asee2/cam1/rgb``` | RGB frames from camera1  |
| ```/asee2/cam2/rgb``` | RGB frames from camera2 |
| ```/asee2/cam1/depth``` | depth frames from camera1  |
| ```/asee2/cam2/depth``` | depth frames from camera2 |
| ```/asee2/merged/pcd``` | combined pointcloud data from both cameras |
| ```/asee2/surface/coeff``` | parameters of the quadratic surface fitted to the combined pointcloud  |
| ```/asee2/normal_vector``` | normal vector at the probe tip, w.r.t camera1 coordinate |


## Citations
