#define PCL_NO_PRECOMPILE

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <thread>
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
#include <pcl/filters/impl/passthrough.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <stereo_depth_correction/calibrationConfig.h>
#include "depth_correction_utils.h"

using namespace std;
using namespace sensor_msgs;

ros::Publisher transformed_cloud_pub, output_image_pub, proj_image_pub, seeded_image_pub;

//Dynamic parameters (data from velo2cam_calibration)
float trans_x;
float trans_y;
float trans_z;
float roll;
float pitch;
float yaw;

int mr; //size of bilateral filter
bool GPU; //To choose multi-threading in GPU

Eigen::Affine3f transform_sv, transform_sc;
cv::Mat velo_im;
float fov_H = 45*CV_PI/180; //biggest fov zed camera has is 87° for VGA resolution. We are considering half of it with a safety margin
unsigned int cores = thread::hardware_concurrency()/2;
vector<thread> threads = vector<thread>(cores);

void bilateralFilter_gpu(const cv::Mat & input, cv::Mat & output, int r);

void callback(const Image::ConstPtr& depth, const PointCloud2::ConstPtr& velo_cloud, const CameraInfo::ConstPtr& info){
	
	ros::Time cycle_start = ros::Time::now();
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

	if (DEBUG) {
  		string ty =  type2str( depth_im.type() );
		ROS_INFO("\n \n data type in depth image is %s %dx%d", ty.c_str(), depth_im.cols, depth_im.rows);
		double min_d, max_d;
		cv::minMaxLoc(depth_im, &min_d, &max_d);
		ROS_INFO("depth map : min is %f and max is %f", min_d, max_d);
	}

	//Replacing NaN values in depth map with zero
	int check_ = 0; //to check the number of infinity pixels
	int check__ = 0; //to check the number of NaN pixels
  	for (int ro = 0; ro < Height; ro++){
   		for (int co = 0; co < Width; co++){
			if (cvIsInf(depth_im.at<float>(ro, co))){
				check_++;
   		  		depth_im.at<float>(ro, co) = 0;
			}
			else if (cvIsNaN(depth_im.at<float>(ro, co))){
				check__++;
   		  		depth_im.at<float>(ro, co) = 0;
			}
   		}
 	}

	if (DEBUG) {
		double min_d, max_d;
		cv::minMaxLoc(depth_im, &min_d, &max_d);
		ROS_INFO("depth map after processing: min is %f and max is %f", min_d, max_d);
		ROS_INFO("Total number of pixels with INF and NaN in depth map were %d and %d", check_, check__);
	}
  	//Pointcloud processing
  	pcl::PointCloud<Velodyne::Point>::Ptr lidar_cloud (new pcl::PointCloud<Velodyne::Point> ());
  	pcl::fromROSMsg(*velo_cloud,*lidar_cloud);

  	//Transforming cloud from velodyne to stereo
  	pcl::PointCloud<Velodyne::Point>::Ptr pre_transformed_cloud (new pcl::PointCloud<Velodyne::Point> ());
  	pcl::transformPointCloud (*lidar_cloud, *pre_transformed_cloud, transform_sv);

	// Filter points using the field of view
	ros::Time start_filter = ros::Time::now();
  	pcl::PointCloud<Velodyne::Point>::Ptr filtered_cloud (new pcl::PointCloud<Velodyne::Point> ());
  	for (pcl::PointCloud<Velodyne::Point>::iterator pot = pre_transformed_cloud->points.begin(); pot < pre_transformed_cloud->points.end(); ++pot){
		if (abs(pot->y / pot->x) < tan(fov_H) && pot->x >0){
			filtered_cloud->push_back(*pot);
		}
	}

	// Remove points which might be occluded by foreground objects due to change of view
  	pcl::PointCloud<Velodyne::Point>::Ptr view_adjusted_cloud (new pcl::PointCloud<Velodyne::Point> ());
  	vector<vector<Velodyne::Point> > rings = Velodyne::getRings(*filtered_cloud);
	for (vector<vector<Velodyne::Point> >::iterator ring = rings.begin(); ring < rings.end(); ++ring){
		Velodyne::Point point_in = (*ring)[0];
        view_adjusted_cloud->push_back(point_in);
		int c = 0;
		int n = 1;
		while (n < (*ring).size()){
			Velodyne::Point current = (*ring)[c];
			Velodyne::Point next = (*ring)[n];
			if ((current.y/current.x) > (next.y/next.x) || ( atan(next.y/next.x) - atan(current.y/current.x) ) > fov_H){
				view_adjusted_cloud->push_back(next);
				c = n++;
			}
			else{
				++n;
			}
		}
	}
	ros::Time finish_filter = ros::Time::now();
	double filter_time = (finish_filter - start_filter).toNSec() * 1e-6;
	if (DEBUG) ROS_INFO_STREAM("Filtering time (ms): " << filter_time);

  	//Transforming cloud from stereo to camera frame
  	pcl::PointCloud<Velodyne::Point>::Ptr transformed_cloud (new pcl::PointCloud<Velodyne::Point> ());
  	pcl::transformPointCloud (*view_adjusted_cloud, *transformed_cloud, transform_sc);

  	//Publish the transformed cloud for checking
	if (DEBUG) {
  		PointCloud2 ros_cloud;
  		pcl::toROSMsg(*view_adjusted_cloud, ros_cloud);
  		ros_cloud.header = velo_cloud->header;
  		transformed_cloud_pub.publish(ros_cloud);
	}

  	//Initiate camera model
  	image_geometry::PinholeCameraModel cam_model; // init cam_model
  	cam_model.fromCameraInfo(*info); // fill cam_model with CameraInfo

  	//Project the points to image using the camera model defined
	ros::Time start_proj = ros::Time::now();
  	cv::Mat mD = cv::Mat::zeros(depth_im.size(), depth_im.type());
  	vector<cv::Point> nonzero;
	vector<cv::Point> nonzero_lr; //for the lowest ring
	vector<cv::Point> nonzero_ur; //for the upper most ring

  	for (pcl::PointCloud<Velodyne::Point>::iterator pt = transformed_cloud->points.begin(); pt < transformed_cloud->points.end(); ++pt){
    	cv::Point3f pt_cv(pt->x, pt->y, pt->z);//init Point3f
    	cv::Point2f uv = cam_model.project3dToPixel(pt_cv); // project 3d point to 2d point
    	int X = (int)round(uv.x); int Y = (int)round(uv.y);
    	if (X < Width && X >= 0 && Y < Height && Y >= 0){
      		mD.at<float>(Y, X) = pt->z;
      		cv::Point* pnt = new cv::Point;
      		pnt->x = X; pnt->y = Y;
			if (pt->ring == 0){
				nonzero_lr.push_back(*pnt);
      			nonzero.push_back(*pnt);
			}
			else if (pt->ring == RINGS_COUNT-1){
				nonzero_ur.push_back(*pnt);
      			nonzero.push_back(*pnt);
			}
			else{
      			nonzero.push_back(*pnt);
			}
      		delete pnt;
    	}
  	}
	ros::Time finish_proj = ros::Time::now();
	double proj_time = (finish_proj - start_proj).toNSec() * 1e-6;
	if (DEBUG) ROS_INFO_STREAM("Projection time (ms): " << proj_time);

	if (DEBUG) {
		double min_p, max_p;
		cv::minMaxLoc(mD, &min_p, &max_p);
		ROS_INFO("Projected image : min is %f and max is %f", min_p, max_p);
	}

	//Publishing projected image
	if (DEBUG) {
  		sensor_msgs::ImagePtr im_msg_1 = cv_bridge::CvImage(std_msgs::Header(), "32FC1", mD).toImageMsg();
  		im_msg_1->header = info->header;
  		proj_image_pub.publish(im_msg_1);
	}

	//Seeding the velodyne depth map from stereo depth map
	vector<int> spread = vector<int>(3); //pixel distribution wrt reference in the neighborhood
	int den; //to define pixel density in neighborhood
	get_spread(spread, den, Height);
	if (DEBUG) ROS_INFO("The spread has been chosen as %d , %d and %d", spread[0], spread[1], spread[2]);
	seedMap(depth_im, mD, nonzero, nonzero_lr, nonzero_ur, Height, Width, spread, den);
	ros::Time finish_seed = ros::Time::now();
	double seed_time = (finish_seed - finish_proj).toNSec() * 1e-6;
	if (DEBUG) ROS_INFO_STREAM("Seeding time (ms): " << seed_time);
	if (DEBUG) {
		double min_p, max_p;
		cv::minMaxLoc(mD, &min_p, &max_p);
		ROS_INFO("Seeded image : min is %f and max is %f", min_p, max_p);
	}

	//Publishing seeded image
	if (DEBUG) {
  		sensor_msgs::ImagePtr im_msg_2 = cv_bridge::CvImage(std_msgs::Header(), "32FC1", mD).toImageMsg();
  		im_msg_2->header = info->header;
  		seeded_image_pub.publish(im_msg_2);
	}

	//Creating dense map
  	velo_im = cv::Mat::zeros(depth_im.size(), depth_im.type());
	ros::Time start_op = ros::Time::now();
	if (GPU){
		bilateralFilter_gpu(mD, velo_im, (mr -1)/2); //Multithreading in GPU
	}
	else {
		for (unsigned int t = 0; t < cores; t++){
			threads[t] = thread(MyBilateralFilter, ref(mD), ref(velo_im), mr, t, cores); //Multithreading in CPU
		}
		for (unsigned int t = 0; t < cores; t++){
			threads[t].join();
		}
	}

	ros::Time finish_op = ros::Time::now();
	double execution_time = (finish_op - start_op).toNSec() * 1e-6;
	if (DEBUG) ROS_INFO_STREAM("Exectution time (ms): " << execution_time);


  	//Count of pixels occupied before and after operation
  	int count_nz = 0;
  	for (int r = 0; r < Height; r++){
    	for (int c = 0; c < Width; c++){
    	  	if (velo_im.at<float>(r, c) > 0){
				count_nz++;
			}
			else{
				velo_im.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();//to replace zeros with NaN
			}
    	}
  	}

	if (DEBUG) {
		double min_v, max_v;
		cv::minMaxLoc(velo_im, &min_v, &max_v);
		ROS_INFO("Total number points projected to camera frame is %ld and after operation is %d ", nonzero.size(), count_nz);
		ROS_INFO("Velodyne image : min is %f and max is %f", min_v, max_v);
	}
	//Publishing processed velo image
  	sensor_msgs::ImagePtr im_msg_3 = cv_bridge::CvImage(std_msgs::Header(), "32FC1", velo_im).toImageMsg();
  	im_msg_3->header = info->header;
  	output_image_pub.publish(im_msg_3);

	ros::Time cycle_finish = ros::Time::now();
	double cycle_time = (cycle_finish - cycle_start).toNSec() * 1e-6;
	ROS_INFO_STREAM("Total Cycle time (ms): " << cycle_time);
}

void param_callback(stereo_depth_correction::calibrationConfig &config, uint32_t level){
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
  	mr = config.mr_BF;
	mr = (mr % 2 == 1) ? mr : mr+1;
  	ROS_INFO("New size of bilateral filter: %d", mr);
	GPU = config.GPU_;
	ROS_INFO("Multi-threading in %s\n", GPU?"GPU":"CPU");

  	//Creating transformation matrix from velodyne to stereo
  	transform_sv = pcl::getTransformation(trans_x, trans_y, trans_z, roll, pitch, yaw);

  	//For debugging
  	Eigen::Matrix4f transform = transform_sv.matrix();
  	ROS_INFO("Velodyne to Stereo Transformation");
  	ROS_INFO("%f, %f, %f, %f",transform(0,0),transform(0,1),transform(0,2),transform(0,3));
 	ROS_INFO("%f, %f, %f, %f",transform(1,0),transform(1,1),transform(1,2),transform(1,3));
  	ROS_INFO("%f, %f, %f, %f",transform(2,0),transform(2,1),transform(2,2),transform(2,3));
  	ROS_INFO("%f, %f, %f, %f",transform(3,0),transform(3,1),transform(3,2),transform(3,3));
}

int main(int argc, char **argv){
  	ros::init(argc, argv, "depth_correction");
  	ros::NodeHandle nh_("~");

  	message_filters::Subscriber<Image> depth_sub;
  	message_filters::Subscriber<PointCloud2> velo_cloud_sub;
  	message_filters::Subscriber<CameraInfo> info_sub;

  	depth_sub.subscribe(nh_, "depth_map", 1);
  	velo_cloud_sub.subscribe(nh_, "point_cloud", 1);
  	info_sub.subscribe(nh_, "camera_info", 1);

  	transformed_cloud_pub = nh_.advertise<PointCloud2> ("transformed_cloud", 1);
  	output_image_pub = nh_.advertise<Image> ("output_image", 1);
  	proj_image_pub = nh_.advertise<Image> ("projected_image", 1);
  	seeded_image_pub = nh_.advertise<Image> ("seeded_image", 1);

  	//Initializing dynamic parameters
  	dynamic_reconfigure::Server<stereo_depth_correction::calibrationConfig> server;
  	dynamic_reconfigure::Server<stereo_depth_correction::calibrationConfig>::CallbackType f;
  	f = boost::bind(param_callback, _1, _2);
  	server.setCallback(f);

  	//Creating transformation matrix from stereo to camera frame
  	transform_sc = pcl::getTransformation(0.0, 0.0, 0.0, 1.57079632679, -1.57079632679, 0.0);

  	while (ros::ok()){
  	//Subscribing to topics
  	typedef message_filters::sync_policies::ApproximateTime<Image, PointCloud2, CameraInfo> MySyncPolicy;
  	message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), depth_sub, velo_cloud_sub, info_sub);
  	sync_.registerCallback(boost::bind(&callback, _1, _2, _3));

  	ros::spin();
  	}
	return 0;
}
