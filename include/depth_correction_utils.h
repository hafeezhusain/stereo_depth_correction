#ifndef depth_correction_utils_H
#define depth_correction_utils_H

#define PCL_NO_PRECOMPILE
#define DEBUG 1

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

static const vector<int> spread = {-10, 10, 20};

void seedMap(cv::Mat& source, cv::Mat& field, vector<cv::Point>& lot, vector<cv::Point>& lot_lr, vector<cv::Point>& lot_ur, int& H){
	for (cv::Point pick : lot){
		float offset = (source.at<float>(pick) != source.at<float>(pick)) ? 0 : (field.at<float>(pick) - source.at<float>(pick));
		for (int q = 0; q < spread.size(); q++){
			float choice = (source.at<float>(pick.y+spread[q], pick.x) != source.at<float>(pick.y+spread[q], pick.x)) ? field.at<float>(pick) :  (source.at<float>(pick.y+spread[q], pick.x) + offset);
			field.at<float>(pick.y+spread[q], pick.x) = (choice >= 0) ? choice : 0;
		}
	}

	for (cv::Point picks : lot_lr){
		float offset_ = (source.at<float>(picks) != source.at<float>(picks)) ? 0 : (field.at<float>(picks) - source.at<float>(picks));
		for (int v = picks.y + 30; v < H; v = v + 10){
			float choice_ = (source.at<float>(v, picks.x) != source.at<float>(v, picks.x)) ? 0 : (source.at<float>(v, picks.x) + offset_);
			field.at<float>(v, picks.x) = (choice_ >= 0) ? choice_ : 0;
		}
	}

	for (cv::Point pic : lot_ur){
		float offset__ = (source.at<float>(pic) != source.at<float>(pic)) ? 0 : (field.at<float>(pic) - source.at<float>(pic));
		for (int b = pic.y - 20; b >= 0; b = b - 10){
			float choice__ = (source.at<float>(b, pic.x) != source.at<float>(b, pic.x)) ? field.at<float>(pic) : (source.at<float>(b, pic.x) + offset__);
			field.at<float>(b, pic.x) = (choice__ >= 0) ? choice__ : 0;
		}
	}
}

void DB_SCAN(vector<tuple<float, int, int>>& S, vector<tuple<float, int, int>>& Si) {
	float eps = 0.1;
	vector<tuple<float, int, int>> S1, S2, S3, S_2;
	S1.push_back(S[0]);
	for(int k = 0; k < S.size() - 1; k++){
		float rk = get<0>(S[k]);
		float rl = get<0>(S[k+1]);
		float DF = abs((rl - rk) / (rl + rk)); //distance function
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
	Si = (S1.size() >= S_2.size()) ? S1 : S_2;
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

void MyBilateralFilter(cv::Mat& source, cv::Mat& target, int mr, unsigned int t, unsigned int c) {
    target = cv::Mat::zeros(source.rows,source.cols,CV_32F);
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
#endif
