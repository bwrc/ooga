//#pragma once
#ifndef PUPIL_ESTIMATOR_H
#define PUPIL_ESTIMATOR_H

#include <opencv2/opencv.hpp>

class PupilEstimator
{
public:

	PupilEstimator();
	~PupilEstimator();

	void Initialize();
	//cv::RotatedRect getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element);
	cv::RotatedRect getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta);
	void getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta, cv::RotatedRect &pupilEllipse);

	void getPupilEllipseThreaded(cv::RotatedRect &pupilEllipse, cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element);

	private:

	bool ITERATE = 1;

	// Initialize variables
	cv::Mat eyeImage_filtered;
	//cv::Mat eyeImage_cropped; can be removed(?)
	cv::Mat eyeImage_closed;
	cv::Mat eyeImage_binary;
	cv::Mat eyeImage_binary_8bit;
	cv::Mat eyeImage_sum;
	//cv::Mat eyeImage_sum_float;
	cv::Mat eyeImage_sum_scaled;
	cv::Mat connComps;
	cv::Mat connComps_32F;
	cv::Mat pupil_blob;
	cv::Mat pupil_blob_8bit;
	cv::Mat connComps_aux;

	std::vector<cv::Point> pupil_contour;
	std::vector<cv::Point2f> pupil_edge;
	double min_val, max_val;
	cv::Point min_loc, max_loc;
	float delta_x = 70;  // zoom area (horizontal) where to search the pupil edge around the pupil center
	float delta_y = 70;  // zoom area (vertical) where to search the pupil edge around the pupil center
	float x_min, x_max, y_min, y_max;

	// these are here for the iteration part
	cv::Mat eyeImage_clone;
	cv::Mat eyeImage_norm;
	//sobel results
	cv::Mat imFilt_up, imFilt_down, imFilt_left, imFilt_right;

};

#endif
