//#pragma once
#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include <thread>
#include <mutex>
#include <vector>

//these were in corneacomputer
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>

#include "SG_common.h"
#include "FrameBinocular.h"

#include "Settings.h"
#include "PerformanceTimer.h"
#include "Cornea_computer.h"
#include "Camera.h"
#include "GlintFinder.h"

// From Ganzheit:
#include "Camera.h"
//#include "getPupilEllipse.h"
#include "PupilEstimator.h"
#include "getPupilEllipsePoints.h"

class EyeTracker
{
public:
	EyeTracker();
	~EyeTracker();	

//	void SetCamFeedSource(FrameSrc f);

	//void InitAndConfigure( TSettings *settings );
	void InitAndConfigure(FrameSrc myEye, std::string CM_fn, std::string glintmodel_fn, std::string K9_matrix_fn);
	virtual void Process(cv::UMat* eyeframe,
			     TTrackingResult* trackres, 
			     cv::Point3d &pupilCenter3D, 
			     cv::Point3d &corneaCenter3D,
			     double &theta);
	void setCropWindowSize(int xmin, int ymin, int width, int height);


	std::vector<cv::Point2d> glintPoints_tmp;  // this can be removed

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
	//double theta;
	double theta_prev;
	double weight;

	float glint_beta;  // The coefficient of the glint likelihood model
	float glint_reg_coef;   // The initial regularization coefficient for the covariance matrix (if zero, might result in non-valid covariance matrix ---> crash)
	bool pupil_iterate;  // Wheter to iterate in the pupil ellipse finder
	float pupil_beta;   // The coefficient of the pupil likelihood model (during iteration)
	double kalman_R_max; // Maximum variance for the observation noise. The larger this is, the less the observations are trusted.

	Camera* eyeCam;
//	double eye_intr[9];// , eye_intr2[9];
//	double eye_dist[5];// , eye_dist2[5];

	cv::Mat K9_matrix; // = cv::Mat(3, 3, CV_64F);  Miika
	cv::Mat A_rot;
	cv::Mat a_tr;
	cv::Mat invA_rot;

	PupilEstimator* pupilestimator;
	cv::Point2d* pupilEllipsePoints;
	cv::Point2d pupilEllipsePoints_prev_eyecam[4];
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > led_pos;
	cv::Point3d Cc3d;
	cv::Point3d Pc3d;

};

#endif
