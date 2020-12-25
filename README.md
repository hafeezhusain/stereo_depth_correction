# stereo_depth_correction

The *stereo_depth_correction* package is meant for improving the depth map produced by a stereo camera with the help of point cloud from a LIDAR. The package is developed in ROS environment.

# Nodes #
## depth_correction ##
The package contains a single node which synchronously subscribes to three topics and publishes four topics.
### Subscribed Topics ###
*depth_map* ([sensor_msgs/Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

&nbsp;&nbsp;&nbsp;&nbsp;Depth image published by the stereo camera (In the reference frame of left camera)

*point_cloud* ([sensor_msgs/PointCloud2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html))

&nbsp;&nbsp;&nbsp;&nbsp;Pointcloud published by LiDAR

*camera_info* ([sensor_msgs/CameraInfo](http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html))

&nbsp;&nbsp;&nbsp;&nbsp;Camera_info parameters corresponding to the depth image, published by the stereo camera

### Published Topics ###
*output_image* ([sensor_msgs/Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

&nbsp;&nbsp;&nbsp;&nbsp;Depth image with improved accuracy (In the reference frame of left stereo camera with same time stamp as input)

**Note**: The additional three topics are published for debugging purposes

*transformed_cloud* ([sensor_msgs/PointCloud2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html))

&nbsp;&nbsp;&nbsp;&nbsp;LiDAR pointcloud transformed to left stereo camera reference frame

*projected_image* ([sensor_msgs/Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

&nbsp;&nbsp;&nbsp;&nbsp;Projection image of LiDAR pointcloud

*seeded_image* ([sensor_msgs/Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

&nbsp;&nbsp;&nbsp;&nbsp;Projection image after seeding process\[1\]

### Parameters ###
The main parameters to be set for the algorithm are the transformation from LiDAR coordinate system to stereo camera coordinate system. The translation parameters (x, y and z) and rotation parameters (roll, pitch and yaw) are to be set. These are dynamically configurable by calling the *rqt_reconfigure* environment as follows:

```rosrun rqt_reconfigure rqt_reconfigure```

It is necessary to perform a calibration of the sensor module in order to obtain the transformation parameters between LiDAR and stereo camera coordinate systems, and it is highly recommended to set the values as default in .cfg file provided in the package.

Next parameter to be set is *mr* which denotes the size of bilateral filter used in the algorithm. It can also be configured dynamically. It is an important parameter which determines the accuracy and density of final output and the computational load required\[1\].

The final boolean parameter *GPU* determines whether the local interpolation of the algorithm to be run in GPU or CPU.

**Note**: Additionally, make sure the correct number of LiDAR laser channels and horizontal field of view of camera are set in the node (Default 16 channels and FOV: 90 degrees).

# Usage #
A sample .launch file is provided in this package. The simplest way to launch the algorithm is by running the ROS node as follows:

```roslaunch stereo_depth_correction depth_correction.launch```


# Citation #
\[1\] Soon.
