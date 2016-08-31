#ifndef GET_PUPIL_CENTER_H
#define GET_PUPIL_CENTER_H

#include <opencv2/opencv.hpp>

cv::Point2d getPupilCenter(cv::Mat eyeImage_filtered);

#endif