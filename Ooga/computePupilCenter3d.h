#ifndef COMPUTE_PUPIL_CENTER_H
#define COMPUTE_PUPIL_CENTER_H

#include <opencv2/opencv.hpp>

cv::Point3d computePupilCenter3d(std::vector<cv::Point3d> pupilEllipsePoints3d, cv::Point3d Cc3d);

#endif