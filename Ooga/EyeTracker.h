//#pragma once
#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include <thread>
#include <mutex>
#include <vector>
#include "SG_common.h"
#include "FrameBinocular.h"

#include "Settings.h"
//#include "Settings.h"
#include "PerformanceTimer.h"

#include "GlintFinder.h"

#include <opencv2\core.hpp>

// From Ganzheit:
#include "Camera.h"

class EyeTracker
{
public:
	EyeTracker();
	~EyeTracker();

//	void SetCamFeedSource(FrameSrc f);

	//void InitAndConfigure( TSettings *settings );
	void InitAndConfigure(FrameSrc myEye, std::string CM_fn, std::string glintmodel_fn);
	virtual void Process(cv::UMat* eyeframe,
							TTrackingResult &trackres, 
							cv::Point3d &pupilCenter3D, 
							cv::Point3d &corneaCenter3D);
	void setCropWindowSize(int xmin, int ymin, int width, int height);

private:
	int framecounter;
	std::mutex m_lock;
	TPerformanceTimer* pt;

	//crop window size
	int cropminX, cropminY, cropsizeX, cropsizeY;

	std::vector<cv::Point2d> glintPoints;
	std::vector<cv::Point2d> glintPoints_prev;
	GlintFinder* glintfinder;

	//initialized temps for holding stage results
	cv::UMat gray;
	cv::UMat cropped;
	cv::UMat opened;
	cv::UMat diffimg; 
	cv::UMat filtered;
	cv::UMat imageDiff;
	cv::UMat previous;

	//filter elements
	cv::Mat glint_element;
	cv::Mat pupil_kernel;
	cv::Mat pupil_kernel2;
	cv::Mat pupil_element;
	cv::Mat glint_kernel;

	//stat variables
	cv::Mat sigmoid_buffer;
	double lambda_ed;
	double alpha_ed;
	double theta;
	double theta_prev;
	double weight;

};

#endif