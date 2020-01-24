#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
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

ros::Publisher transformed_cloud_pub, velo_image_pub;

//Dynamic parameters (data from velo2cam_calibration)
bool inverse_transform;
float trans_x;
float trans_y;
float trans_z;
float roll;
float pitch;
float yaw;

int mr; //size of bilateral filter

//Image titles  
int fontFace = cv::FONT_HERSHEY_DUPLEX;
string title_dep = "ORIGINAL DEPTH MAP";
string title_lid = "PROJECTED LIDAR MAP";

Eigen::Affine3f transform_sv, transform_vs, transform_sc, transformation;
cv::Mat velo_im;

//Function to identify data type in depth image
string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
    	default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

float distance(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

void applyBilateralFilter(cv::Mat& source, cv::Mat& target, int x, int y, int mr) {
    float r_c = 0;
    float Ws = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = (mr - 1) / 2;
	vector<tuple<float, int, int>> nz;
    for(int i = 0; i < mr; i++) {
        for(int j = 0; j < mr; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
			if (source.at<float>(neighbor_y, neighbor_x) != 0){
				nz.push_back(make_tuple(source.at<float>(neighbor_y, neighbor_x), neighbor_x, neighbor_y));
			}
		}
	}
	if (nz.empty()){
		target.at<float>(y, x) = r_c;
	}
	else{
	    sort(nz.begin(), nz.end());
		float r0 = (source.at<float>(y, x) != 0) ? source.at<float>(y, x) : get<0>(nz[0]);
		for(tuple<float, int, int> P : nz){
			float ri = get<0>(P);
			int xi = get<1>(P);
			int yi = get<2>(P);
            float gs = 1 / (1 + distance(x, y, xi, yi));
			float gr = 1 / (1 + abs(r0 - ri));
            float w = gs * gr;
            r_c = r_c + ri * w;
            Ws = Ws + w;
        }
    	r_c = r_c / Ws;
    	target.at<float>(y, x) = r_c;
	}
}

void MyBilateralFilter(cv::Mat& source, cv::Mat& target, int mr) {
    target = cv::Mat::zeros(source.rows,source.cols,CV_32F);
    int width = source.cols;
    int height = source.rows;
    int half = (mr - 1) / 2;
    for(int i = half; i < width - half; i++) {
        for(int j = half; j < height - half; j++) {
            applyBilateralFilter(source, target, i, j, mr);
        }
    }
}

void SparseToDense(const cv::Mat& mX, const cv::Mat& mY, const cv::Mat& mD, int grid, cv::Mat &out){
    int ng = 2*grid+1;
    int H = mD.rows;
    int W = mD.cols;
    cv::Mat KmX(H-2*grid, W-2*grid, mD.type());
    cv::Mat KmY(H-2*grid, W-2*grid, mD.type());
    cv::Mat KmD(H-2*grid, W-2*grid, mD.type());
    cv::Mat s(H-2*grid, W-2*grid, mD.type());
    cv::Mat S = cv::Mat::zeros(H-2*grid, W-2*grid, mD.type());
    cv::Mat Y = cv::Mat::zeros(H-2*grid, W-2*grid, mD.type());

    for (int i = 0; i < ng; i++){
    	for (int j = 0; j < ng; j++){
      	KmX = mX(cv::Range(i, H-ng+1+i), cv::Range(j, W-ng+1+j))-(float)grid-1+(float)i;
      	KmY = mY(cv::Range(i, H-ng+1+i), cv::Range(j, W-ng+1+j))-(float)grid-1+(float)j;
      	KmD = mD(cv::Range(i, H-ng+1+i), cv::Range(j, W-ng+1+j));
      	for (int p = 0; p < s.rows; p++){
        	for (int q = 0; q < s.cols; q++){
          	if (KmX.at<float>(p, q) > 500 || KmY.at<float>(p, q) > 500)
            	s.at<float>(p, q) = 0;
          	else
            	s.at<float>(p, q) = 1/sqrt(KmX.at<float>(p, q)*KmX.at<float>(p, q) + KmY.at<float>(p, q)*KmY.at<float>(p, q));
        	}
      	}
      	Y += s.mul(KmD);
      	S += s;
    	}
  	}

  	for (int l = grid; l < H-grid; l++){
    	for (int k = grid; k < W-grid; k++){
      	if (S.at<float>(l-grid, k-grid) == 0)
        	S.at<float>(l-grid, k-grid) = 1;
      	out.at<float>(l, k) = Y.at<float>(l-grid, k-grid) / S.at<float>(l-grid, k-grid);
    	}
  	}
}

void callback(const Image::ConstPtr& depth, const PointCloud2::ConstPtr& velo_cloud, const CameraInfo::ConstPtr& info){ 
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
  	string ty =  type2str( depth_im.type() );
//	ROS_INFO("data type in depth image is %s %dx%d \n", ty.c_str(), depth_im.cols, depth_im.rows);

/*	//displaying depth image
  	cv::Mat depth_im1;
  	cv::resize(depth_im, depth_im1, cv::Size(), 0.5, 0.5);
  	cv::putText(depth_im1, title_dep, cv::Point(depth_im1.cols-500*Width/1920, depth_im1.rows-50*Width/1920), fontFace, 1, 0, 2, cv::LINE_AA);
  	cv::imshow("raw_depth", depth_im1);
  	cv::waitKey(100);*/

  	//Pointcloud processing
  	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  	pcl::fromROSMsg(*velo_cloud,*lidar_cloud);

  	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  	// Filter the points in positive x
  	pcl::PassThrough<pcl::PointXYZ> pass_x;
  	pass_x.setInputCloud (lidar_cloud);
  	pass_x.setFilterFieldName ("x");
  	pass_x.setFilterLimits (0, 20); //for debugging. actual limit should be 100 or even 20 since that's the limit for zed
  	pass_x.filter (*filtered_cloud);

  	//Transforming cloud from velodyne to stereo
  	pcl::PointCloud<pcl::PointXYZ>::Ptr pre_transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  	pcl::transformPointCloud (*filtered_cloud, *pre_transformed_cloud, transformation);

  	//Transforming cloud from stereo to camera frame
  	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  	pcl::transformPointCloud (*pre_transformed_cloud, *transformed_cloud, transform_sc);

  	//Publish the transformed cloud for checking
  	PointCloud2 ros_cloud;
  	pcl::toROSMsg(*transformed_cloud, ros_cloud);
  	ros_cloud.header = velo_cloud->header;
  	transformed_cloud_pub.publish(ros_cloud);

  	//Initiate camera model
  	image_geometry::PinholeCameraModel cam_model; // init cam_model
  	cam_model.fromCameraInfo(*info); // fill cam_model with CameraInfo

  	//Project the points to image using the camera model defined
  	cv::Mat mD = cv::Mat::zeros(depth_im.size(), depth_im.type());
  	cv::Mat mX(depth_im.size(), depth_im.type(), cv::Scalar(1000));
  	cv::Mat mY(depth_im.size(), depth_im.type(), cv::Scalar(1000));
  	vector<cv::Point> nonzero; int Ymin = Height; int Ymax = 0;

  	for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = transformed_cloud->points.begin(); pt < transformed_cloud->points.end(); ++pt){
    	cv::Point3f pt_cv(pt->x, pt->y, pt->z);//init Point3f
    	cv::Point2f uv = cam_model.project3dToPixel(pt_cv); // project 3d point to 2d point
    	int X = (int)round(uv.x); int Y = (int)round(uv.y);
    	if (X < Width && X >= 0 && Y < Height && Y >= 0){
      		mD.at<float>(Y, X) = pt->z;
      		mX.at<float>(Y, X) = uv.x - (float)X;
      		mY.at<float>(Y, X) = uv.y - (float)Y;
      		Ymin = (Y < Ymin) ? Y :Ymin;
      		Ymax = (Y > Ymax) ? Y :Ymax;
      		cv::Point* pnt = new cv::Point;
      		pnt->x = X; pnt->y = Y;
      		nonzero.push_back(*pnt);
      		delete pnt;
    	}
  	}

	cv::Mat mD_roi(mD, cv::Rect(0, Ymin, Width, (Ymax - Ymin + 1)));
  	velo_im = cv::Mat::zeros(depth_im.size(), depth_im.type());
	cv::Mat velo_im_roi(velo_im, cv::Rect(0, Ymin, Width, (Ymax - Ymin + 1)));
//	SparseToDense(mX, mY, mD, 4, velo_im);
	MyBilateralFilter(mD_roi, velo_im_roi, mr);

  	//Count of pixels occupied before and after operation
  	vector<cv::Point> nonzero2;  
  	for (int r = 0; r < Height; r++){
    	for (int c = 0; c < Width; c++){
      		if (velo_im.at<float>(r, c) > 0){
        	cv::Point* pnt = new cv::Point;
        	pnt->x = c; pnt->y = r;
        	nonzero2.push_back(*pnt);
        	delete pnt;
      		}
    	}
  	}
	ROS_INFO("Total number points projected to camera frame is %ld and after operation is %ld", nonzero.size(), nonzero2.size());
/*
  	//Interpolation attempt
  	for (cv::Point pint : nonzero){
      	float depth = mD.at<float>(pint);
      	for (int j = 0; j < 11; j++){
        	if (pint.y < (Height - 6))
          	mD.at<float>((pint.y-5+j),pint.x) = depth;
      	}
  	}
*/

	//Publishing projected velo image
  	sensor_msgs::ImagePtr im_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", velo_im).toImageMsg();
  	im_msg->header = info->header;
  	velo_image_pub.publish(im_msg);

  	//displaying velo image
  	cv::Mat velo_im1;
  	cv::resize(velo_im, velo_im1, cv::Size(), 0.5, 0.5);
//	cv::putText(velo_im1, title_lid, cv::Point((velo_im1.cols)/2-200*Width/1920, velo_im1.rows-50*Width/1920), fontFace, 1, 1, 2, cv::LINE_AA);
  	cv::imshow("velo_depth", velo_im1);
  	cv::waitKey(100);

  	cv::Mat velo_im2;
  	cv::resize(mD, velo_im2, cv::Size(), 0.5, 0.5);
//	cv::putText(velo_im2, title_lid, cv::Point((velo_im2.cols)/2-200*Width/1920, velo_im2.rows-50*Width/1920), fontFace, 1, 1, 2, cv::LINE_AA);
  	cv::imshow("raw_depth", velo_im2);
  	cv::waitKey(100);

/*	//displaying both images for comparison
  	cv::Mat big_im = cv::Mat::zeros(depth_im.size(), CV_32FC3);;
//	cv::hconcat(depth_im, velo_im, big_im);
  	for (int r = 0; r < Height; r++){
    	for (int c = 0; c < Width; c++){
      	if (velo_im.at<float>(r, c) > 0)
        	big_im.at<cv::Vec3f>(r, c)[0] = velo_im.at<float>(r, c);
      	else
        	big_im.at<cv::Vec3f>(r, c)[1] = depth_im.at<float>(r, c);
    	}
  	}
  	cv::resize(big_im, big_im, cv::Size(), 0.5, 0.5);
  	cv::imshow("Depth_Comparison", big_im);
  	cv::waitKey(100);*/

/*	//Benchmarking
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
  	ROS_INFO("New yaw angle: %f", yaw);
  	mr = config.mr_;
  	ROS_INFO("New size of bilateral filter: %d\n", mr);

  	//Creating transformation matrix from velodyne to stereo
  	transform_sv = pcl::getTransformation(trans_x, trans_y, trans_z, roll, pitch, yaw);
  	Eigen::Matrix3f R = transform_sv.rotation();
  	Eigen::VectorXf b = transform_sv.translation();
  	Eigen::Matrix4f inv; inv << R.inverse(), -R.inverse()*b,
                              0, 0, 0, 1;
  	transform_vs.matrix() = inv; //the inverse transformation

  	if (inverse_transform)
    	transformation.matrix() = inv;
  	else
    	transformation.matrix() = transform_sv.matrix();

  	//For debugging
  	Eigen::Matrix4f transform = transformation.matrix();
  	ROS_INFO("Velodyne to Stereo Transformation");
  	ROS_INFO("%f, %f, %f, %f",transform(0,0),transform(0,1),transform(0,2),transform(0,3));
 	 ROS_INFO("%f, %f, %f, %f",transform(1,0),transform(1,1),transform(1,2),transform(1,3));
  	ROS_INFO("%f, %f, %f, %f",transform(2,0),transform(2,1),transform(2,2),transform(2,3));
  	ROS_INFO("%f, %f, %f, %f",transform(3,0),transform(3,1),transform(3,2),transform(3,3));
}

int main(int argc, char **argv){
  	ros::init(argc, argv, "depth_benchmarking");
  	ros::NodeHandle nh_("~");
  	cv::namedWindow("raw_depth", cv::WINDOW_NORMAL);
	cv::namedWindow("velo_depth", cv::WINDOW_NORMAL);
//	cv::namedWindow("Depth_Comparison", cv::WINDOW_NORMAL);

  	message_filters::Subscriber<Image> depth_sub;
  	message_filters::Subscriber<PointCloud2> velo_cloud_sub;
  	message_filters::Subscriber<CameraInfo> info_sub;

  	depth_sub.subscribe(nh_, "depth_map", 1);
  	velo_cloud_sub.subscribe(nh_, "point_cloud", 1);
  	info_sub.subscribe(nh_, "camera_info", 1);

  	transformed_cloud_pub = nh_.advertise<PointCloud2> ("transformed_cloud", 1);
  	velo_image_pub = nh_.advertise<Image> ("velo_image", 1);

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

  	ros::waitForShutdown();
  	cv::destroyWindow("raw_depth");
	cv::destroyWindow("velo_depth");
//	cv::destroyWindow("Depth_Comparison");
}
