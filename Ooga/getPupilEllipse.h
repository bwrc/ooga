#ifndef GET_PUPIL_ELLIPSE_H
#define GET_PUPIL_ELLIPSE_H

#include <opencv2/opencv.hpp>

//cv::RotatedRect getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element);
//cv::RotatedRect getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta);
void getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta, cv::RotatedRect* pupilEllipse);


#endif