#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <stereo_depth_correction/calibrationConfig.h>

using namespace std;
using namespace sensor_msgs;

ros::Publisher transformed_cloud_pub;

//Dynamic parameters (data from velo2cam_calibration)
bool inverse_transform;
float trans_x;
float trans_y;
float trans_z;
float roll;
float pitch;
float yaw;

//Image titles  
int fontFace = cv::FONT_HERSHEY_DUPLEX;
string title_dep = "ORIGINAL DEPTH MAP";
string title_lid = "PROJECTED LIDAR MAP";

Eigen::Affine3f transform_sv, transform_vs, transform_sc;

void callback(const Image::ConstPtr& depth, const PointCloud2::ConstPtr& velo_cloud){  
  //Depth Image
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(depth);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat depth_im;
  depth_im = cv_ptr->image;
  int Height = depth_im.rows;
  int Width = depth_im.cols;

  //image title
  cv::putText(depth_im, title_dep, cv::Point(Width-500*Width/1920, Height-50*Width/1920), fontFace, 1, 0, 4, cv::LINE_AA);

  //displaying depth image
  cv::resize(depth_im, depth_im, cv::Size(), 0.5, 0.5);
  cv::imshow("raw_depth", depth_im);
  cv::waitKey(100);

  //Pointcloud processing
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::fromROSMsg(*velo_cloud,*lidar_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  // Filter the points in positive x
  pcl::PassThrough<pcl::PointXYZ> pass_x;
  pass_x.setInputCloud (lidar_cloud);
  pass_x.setFilterFieldName ("x");
  pass_x.setFilterLimits (0, 100);
  pass_x.filter (*filtered_cloud);

  //Transforming cloud from velodyne to stereo
  pcl::PointCloud<pcl::PointXYZ>::Ptr pre_transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  if (inverse_transform){
    pcl::transformPointCloud (*filtered_cloud, *pre_transformed_cloud, transform_vs);}
  else{
    pcl::transformPointCloud (*filtered_cloud, *pre_transformed_cloud, transform_sv);}

  //Transforming cloud from stereo to camera frame
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud (*pre_transformed_cloud, *transformed_cloud, transform_sc);

  //Publish the transformed cloud for checking
  PointCloud2 ros_cloud;
  pcl::toROSMsg(*transformed_cloud, ros_cloud);
  ros_cloud.header = velo_cloud->header;
  transformed_cloud_pub.publish(ros_cloud);

  //Define camerainfo data (from stereo calibration)
  CameraInfo cam_info;
  cam_info.header = depth->header;
  cam_info.height = depth->height;
  cam_info.width = depth->width;
  cam_info.distortion_model = "plumb_bob";
  cam_info.D = {-0.1851100564408104, 0.037527799483959706, -0.0019125212676454016, -0.0004984141236978598, 0.0};
  cam_info.K = {1395.272087011584, 0.0, 964.7003601191602, 0.0, 1393.1819642050018, 563.7474333489621, 0.0, 0.0, 1.0};
  cam_info.R = {0.9948977527414851, 0.005511766585136956, -0.1007376891687207, -0.0053894429534723344, 0.9999843721485911, 0.0014863926933930333, 0.10074430750466294, -0.0009358887213086368, 0.994911909978822};
  cam_info.P = {1432.3701218923425, 0.0, 1167.794288635254, 0.0, 0.0, 1432.3701218923425, 537.1868515014648, 0.0, 0.0, 0.0, 1.0, 0.0};
  cam_info.binning_x = 0;
  cam_info.binning_y = 0;
  cam_info.roi.x_offset = 0;
  cam_info.roi.y_offset = 0;
  cam_info.roi.height = 0;
  cam_info.roi.width = 0;
  cam_info.roi.do_rectify = 0;

  //Initiate camera model
  image_geometry::PinholeCameraModel cam_model; // init cam_model
  cam_model.fromCameraInfo(cam_info); // fill cam_model with CameraInfo

  //Project the points to image using the camera model defined
  cv::Mat velo_im;
  velo_im = cv::Mat::zeros(Height, Width, CV_32F);

  for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = transformed_cloud->points.begin(); pt < transformed_cloud->points.end(); ++pt){
    cv::Point3f pt_cv(pt->x, pt->y, pt->z);//init Point3f
    cv::Point2i uv = cam_model.project3dToPixel(pt_cv); // project 3d point to 2d point
    if (uv.x < Width && uv.x >= 0 && uv.y < Height && uv.y >= 0)
      velo_im.at<float>(uv.y,uv.x) = pt->z;
  }
/*
  //Interpolation attempt - 1
  vector<int> nz;
  for (int c = 0; c < Width; c++){
    for (int r = 0; r < Height; r++){
      if (velo_im.at<float>(r, c) > 0)
        nz.push_back(r);
    }
    for (int i = 1; i < nz.size(); i++){
      int count = nz[i] - nz[i-1];
      float start = velo_im.at<float>(nz[i-1], c);
      float end = velo_im.at<float>(nz[i], c);
      if ((end-start) < 1.0){
        for (int j = 1; j < count; j++)
          velo_im.at<float>((nz[i-1]+j),c) = start + (end - start)*j/count;
      }
    }
    nz.clear();
  }
*/
  //Interpolation attempt - 2
  vector<int> nz;
  for (int c = 0; c < Width; c++){
    for (int r = 0; r < Height; r++){
      if (velo_im.at<float>(r, c) > 0)
        nz.push_back(r);
    }
    for (int i = 0; i < nz.size(); i++){
      float depth = velo_im.at<float>(nz[i], c);
      for (int j = 0; j < 11; j++){
        if (nz[i] < (Height - 6))
          velo_im.at<float>((nz[i]-5+j),c) = depth;
      }
    }
    nz.clear();
  }

  //image title
  cv::putText(velo_im, title_lid, cv::Point(Width/2-500*Width/1920, Height-50*Width/1920), fontFace, 1, 1, 4, cv::LINE_AA);

  //displaying velo image 
  cv::resize(velo_im, velo_im, cv::Size(), 0.5, 0.5);
  cv::imshow("velo_depth", velo_im);
  cv::waitKey(100);

/*  //displaying both images for comparison
  cv::Mat big_im;
  cv::hconcat(depth_im, velo_im, big_im);
  cv::resize(big_im, big_im, cv::Size(), 0.25, 0.25);
  cv::imshow("Depth_Comparison", big_im);
  cv::waitKey(100);*/

/*  //Benchmarking
  cv::Mat bm_im;
  bm_im = cv::Mat::zeros(Height, Width, CV_32F);
  for (int ii = 0; ii < Height; ii++){
    for (int jj = 0; jj < Width; jj++){
      if (velo_im.at<float>(ii, jj) > 0){
        bm_im.at<cv::Vec3f>(ii, jj)[0] = (velo_im.at<float>(ii, jj) - depth_im.at<float>(ii, jj));
      }
    }
  }

  //displaying difference image 
  cv::resize(bm_im, bm_im, cv::Size(), 0.5, 0.5);
  cv::imshow("Depth_Comparison", bm_im);
  cv::waitKey(100);*/
}

void param_callback(stereo_depth_correction::calibrationConfig &config, uint32_t level){
  inverse_transform = config.inverse_transform_;
  ROS_INFO("Use inverse transform: %s", inverse_transform ? "true" : "false");
  trans_x = (float)config.trans_x_;
  ROS_INFO("New x coordinate of translation: %f", trans_x);
  trans_y = (float)config.trans_y_;
  ROS_INFO("New y coordinate of translation: %f", trans_y);
  trans_z = (float)config.trans_z_;
  ROS_INFO("New z coordinate of translation: %f", trans_z);
  roll = (float)config.roll_;
  ROS_INFO("New roll angle: %f", roll);
  pitch = (float)config.pitch_;
  ROS_INFO("New pitch angle: %f", pitch);
  yaw = (float)config.yaw_;
  ROS_INFO("New yaw angle: %f\n", yaw);

  //Creating transformation matrix from velodyne to stereo
  transform_sv = pcl::getTransformation(trans_x, trans_y, trans_z, roll, pitch, yaw);
  Eigen::Matrix3f R = transform_sv.rotation();
  Eigen::VectorXf b = transform_sv.translation();
  Eigen::Matrix4f inv; inv << R.inverse(), -R.inverse()*b,
                              0, 0, 0, 1;
  transform_vs.matrix() = inv; //the inverse transformation
/*
  ROS_INFO("Velodyne to Stereo Transformation");
  ROS_INFO("x=%f, y=%f, z=%f, roll=%f, pitch=%f, yaw=%f",trans_x, trans_y, trans_z, roll, pitch, yaw);
  ROS_INFO("%f, %f, %f, %f",inv(0,0),inv(0,1),inv(0,2),inv(0,3));
  ROS_INFO("%f, %f, %f, %f",inv(1,0),inv(1,1),inv(1,2),inv(1,3));
  ROS_INFO("%f, %f, %f, %f",inv(2,0),inv(2,1),inv(2,2),inv(2,3));
  ROS_INFO("%f, %f, %f, %f",inv(3,0),inv(3,1),inv(3,2),inv(3,3));
*/
}

int main(int argc, char **argv){
  ros::init(argc, argv, "depth_benchmarking");
  ros::NodeHandle nh_("~");
  cv::namedWindow("raw_depth", cv::WINDOW_NORMAL);
  cv::namedWindow("velo_depth", cv::WINDOW_NORMAL);
//  cv::namedWindow("Depth_Comparison", cv::WINDOW_NORMAL);

  message_filters::Subscriber<Image> depth_sub;
  message_filters::Subscriber<PointCloud2> velo_cloud_sub;

  depth_sub.subscribe(nh_, "depth_map", 1);
  velo_cloud_sub.subscribe(nh_, "point_cloud", 1);

  transformed_cloud_pub = nh_.advertise<PointCloud2> ("transformed_cloud", 1);

  //Initializing dynamic parameters
  dynamic_reconfigure::Server<stereo_depth_correction::calibrationConfig> server;
  dynamic_reconfigure::Server<stereo_depth_correction::calibrationConfig>::CallbackType f;
  f = boost::bind(param_callback, _1, _2);
  server.setCallback(f);

  //Creating transformation matrix from stereo to camera frame
  transform_sc = pcl::getTransformation(0.0, 0.0, 0.0, 1.57079632679, -1.57079632679, 0.0);

  while (ros::ok()){
  //Subscribing to topics
  typedef message_filters::sync_policies::ApproximateTime<Image, PointCloud2> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), depth_sub, velo_cloud_sub);
  sync_.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();
  }

  ros::waitForShutdown();
  cv::destroyWindow("raw_depth");
  cv::destroyWindow("velo_depth");
//  cv::destroyWindow("Depth_Comparison");
}
