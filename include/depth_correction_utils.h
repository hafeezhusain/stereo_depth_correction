#ifndef depth_correction_utils_H
#define depth_correction_utils_H

#define PCL_NO_PRECOMPILE
#define DEBUG 0

#include <vector>
#include <algorithm>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace sensor_msgs;

static const int RINGS_COUNT = 16; // number of laser channels

namespace Velodyne {
	struct Point
  	{
    	PCL_ADD_POINT4D; // preferred way of adding a XYZ+padding
    	uint16_t ring; ///< laser ring number
    	EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
  	}EIGEN_ALIGN16; // enforce SSE padding for correct memory alignment

	vector<vector<Point> > getRings(pcl::PointCloud<Velodyne::Point> & pc)
  	{
    	vector<vector<Point> > rings(RINGS_COUNT);  //a vector of total 16 vectors one for each ring
    	for (pcl::PointCloud<Point>::iterator pt = pc.points.begin(); pt < pc.points.end(); pt++)
    	{
    		rings[pt->ring].push_back(*pt);
    	}
    	return rings;
  	}
}

POINT_CLOUD_REGISTER_POINT_STRUCT(Velodyne::Point, (float, x, x) (float, y, y) (float, z, z) (uint16_t, ring, ring));

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

//To choose the avg of neighbouring values in the short distance region
float avg_choice(cv::Mat& source, int& x, int& y, int& W){
	float avg = 0;
	int coun = 0;
	for (int m = -4; m < 5; m++){
		while(x+m >= 0 && x+m < W){
			if (source.at<float>(y, x+m) != 0){
				avg = avg + source.at<float>(y, x+m);
				coun++;
			}
			break;
		}
	}
	avg = (coun == 0) ? avg : float(avg/(float)coun);
	return avg;
}

//static const vector<int> spread = {-10, 10, 20}; //pixel distribution wrt reference in the neighborhood (try and find the best choice for each resolution)
void get_spread(vector<int>& spread_, int& density_, int& height){
	switch (height){
		case 1242: spread_ = {-10, 10, 20};
				   density_ = 10;
				   break;
		case 1080: spread_ = {-10, 10, 20};
				   density_ = 10;
				   break;
		case 720:  spread_ = {-5, 5, 10};
				   density_ = 5;
				   break;
		case 376:  spread_ = {-2, 2, 4};
				   density_ = 3;
				   break;
		default:   ROS_INFO("The image resolution is not predefined. Please check get_spread function");
				   break;
	}
}

//Seeding process
void seedMap(cv::Mat& source, cv::Mat& field, vector<cv::Point>& lot, vector<cv::Point>& lot_lr, vector<cv::Point>& lot_ur, int& H, int& W, vector<int>& spread, int& den){
	for (cv::Point pick : lot){
		float offset = (source.at<float>(pick) == 0) ? 0 : (field.at<float>(pick) - source.at<float>(pick));
		for (int q = 0; q < spread.size(); q++){
			float choice = (source.at<float>(pick.y+spread[q], pick.x) == 0 || offset == 0) ? field.at<float>(pick) :  (source.at<float>(pick.y+spread[q], pick.x) + offset);
			field.at<float>(pick.y+spread[q], pick.x) = (choice >= 0) ? choice : field.at<float>(pick);
		}
	}

	for (cv::Point picks : lot_lr){
		float offset_ = (source.at<float>(picks) == 0) ? 0 : (field.at<float>(picks) - source.at<float>(picks));
		for (int v = picks.y + (3 * den); v < H; v = v + den){
			float choice_ = (source.at<float>(v, picks.x) == 0) ? (avg_choice(source, picks.x, v, W) + offset_) : (source.at<float>(v, picks.x) + offset_);
			field.at<float>(v, picks.x) = (choice_ >= 0) ? choice_ : 0;
		}
	}

	for (cv::Point pic : lot_ur){
		float offset__ = (source.at<float>(pic) == 0) ? 0 : (field.at<float>(pic) - source.at<float>(pic));
		for (int b = pic.y - (2 * den); b >= 0; b = b - den){
			float choice__ = (source.at<float>(b, pic.x) == 0 || offset__ == 0) ? field.at<float>(pic) : (source.at<float>(b, pic.x) + offset__);
			field.at<float>(b, pic.x) = (choice__ >= 0) ? choice__ : 0;
		}
	}
}

//Clustering algorithm
void DB_SCAN(vector<tuple<float, int, int>>& S, vector<tuple<float, int, int>>& Si) {
	float eps = 0.1; //adjustable parameter to control the clustering
	vector<tuple<float, int, int>> S1, S2, S3, S_2;
	S1.push_back(S[0]);
	for(int k = 0; k < S.size() - 1; k++){
		float rk = get<0>(S[k]);
		float rl = get<0>(S[k+1]);
		float DF = abs((rl - rk) / (rl + rk)); //distance function of DBSCAN
		if (DF > eps && S2.empty()){
			S2.push_back(S[k+1]);
		}
		else if (DF > eps){
			S3.push_back(S[k+1]);
		}
		else if (S2.empty()){
			S1.push_back(S[k+1]);
		}
		else {
			S2.push_back(S[k+1]);
		}
	}
	if (S3.empty()){
		S_2 = S2;
	}
	else{
		S_2 = (S2.size() >= S3.size()) ? S2 : S3;
	}
	Si = (S1.size() >= S_2.size()) ? S1 : S_2; //choice of significant cluster
}

//Bilateral filter
void applyBilateralFilter(cv::Mat& source, cv::Mat& target, int x, int y, int mr) {
    float r_c = 0;
    float Ws = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = (mr - 1) / 2;
	vector<tuple<float, int, int>> nz;
    for(int i = 0; i < mr; i++) {
        for(int j = 0; j < mr; j++) {
            neighbor_x = x - (half - j);
            neighbor_y = y - (half - i);
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
		vector<tuple<float, int, int>> Si;
		DB_SCAN(nz, Si);
		float r0 = (source.at<float>(y, x) != 0) ? source.at<float>(y, x) : get<0>(Si[0]);
		for(tuple<float, int, int> P : Si){
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

//Calling bilateral fiter (multithreaded)
void MyBilateralFilter(cv::Mat& source, cv::Mat& target, int mr, unsigned int t, unsigned int c) {
    int width = source.cols;
    int height = source.rows;
    int half = (mr - 1) / 2;
    int start  = half + t*(height - 2*half)/c;
    int finish  = half + (t+1)*(height - 2*half)/c;
    for(int i = start; i < finish; i++) {
        for(int j = half; j < width - half; j++) {
            applyBilateralFilter(source, target, j, i, mr);
        }
    }
}

//Calling bilateral filter (single threaded)
void OldBilateralFilter(cv::Mat& source, cv::Mat& target, int mr) {
    int width = source.cols;
    int height = source.rows;
    int half = (mr - 1) / 2;
    for(int i = half; i < height - half; i++) {
        for(int j = half; j < width - half; j++) {
            applyBilateralFilter(source, target, j, i, mr);
        }
    }
}
#endif
