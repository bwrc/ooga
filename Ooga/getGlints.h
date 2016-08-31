#include <opencv2/opencv.hpp>

const double PI = 3.1415926535898;

cv::Mat log_mvnpdf(cv::Mat x, cv::Mat mu, cv::Mat C);

std::vector<cv::Point2d> getGlints(cv::Mat eyeImage_diff, cv::Point2d pupil_center, std::vector<cv::Point2d> glintPoints_prev, \
				   float theta, float MU_X[], float MU_Y[], cv::Mat CM, cv::Mat glint_kernel, double *score);

void getGlintsThreaded(std::vector<cv::Point2d> &glintPoints, 
	cv::Mat eyeImage_diff, 
	cv::Point2d pupil_center, 
	std::vector<cv::Point2d> glintPoints_prev, 
	float theta, 
	float MU_X[], 
	float MU_Y[], 
	cv::Mat CM, 
	cv::Mat glint_kernel, 
	double *score);
