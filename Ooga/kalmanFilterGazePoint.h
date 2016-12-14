#include <opencv2/opencv.hpp>

//void kalmanFilterGazePoint(double *pog_x, double *pog_y, float theta);
void kalmanFilterGazePoint(cv::Point2d const loc_meas, cv::Point2d const velo_meas, cv::Mat *param_est, cv::Mat *P_est, double loc_variance);
