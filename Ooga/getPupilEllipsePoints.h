#ifndef GET_PUPIL_ELLIPSE_POINTS_H
#define GET_PUPIL_ELLIPSE_POINTS_H

#include <opencv2/opencv.hpp>

cv::Point2d* getPupilEllipsePoints(cv::RotatedRect pupilEllipse, cv::Point2d *pupilEllipsePoints_prev, double theta);

#endif