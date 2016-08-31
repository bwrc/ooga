//#pragma once
#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include <thread>
#include <mutex>
#include <vector>
#include "SG_common.h"
#include "FrameBinocular.h"

//#include "Settings.h"
//#include "Settings.h"

#include "PupilEstimator.h"

#include <opencv2\core.hpp>

// From Ganzheit:
#include "Camera.h"
//excluded for gsltest 
#include "Cornea_computer.h"
//apparently eigen headers were included through cc header?!
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

class EyeTracker
{
public:
	EyeTracker();
	~EyeTracker();

	void SetCamFeedSource(FrameSrc f);

	//void InitAndConfigure( TSettings *settings );
	void InitAndConfigure(); // Doesn't read settings now. t: Miika
	virtual void Process(const cv::UMat* eyeframe, TTrackingResult &trackres, cv::Point3d &pupilCenter3D, cv::Point3d &corneaCenter3D);

	cv::Mat K9_matrix;
	//Eigen::Matrix4d A;//(4,4);

private:
	int framecounter;
	FrameSrc src;
	std::mutex m_lock;

	PupilEstimator *pupilest;

	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > led_pos;
	int N_leds = 6;     // Number of leds

	std::vector<cv::Point3d> target3d_points;
	std::vector<cv::Point3d> Cc_points;
	std::vector<cv::Point3d> L_points;
	int N_cal_points = 1;

	bool first_time = true;
	double theta = -1;
	double theta_prev = -1;
	cv::Point2d pupil_center;


	// TODO: Read these from confix file
	int Svert = 480;  // Original image size, vertical
	int Shori = 640;  // Original image size, horizontal
	double weight = 0.3;  // The weight for computing theta  (low weight ---> slower increase of filtering after eye movements)
	double beta_ed = -0.05;  // was -0.03
	double lambda_ed = 200;  // was 300
	float glint_model_MEAN_X[6];// = {-1.0330,  -28.0891,  -26.5871,   -4.4676,   26.1732,   34.0037};
	float glint_model_MEAN_Y[6];// = {-21.3168,  -11.5037,    1.3535,   16.4336,   16.5437,   -1.5103};
	cv::Mat glint_model_CM; //(N_leds*2, N_leds*2, CV_32F);  // to init

	cv::Mat eyeImage_RGB;
	cv::Mat eyeImage_unflipped;
	cv::Mat eyeImage;
	cv::Mat eyeImage_filtered;
	cv::Mat eyeImage_prev;
	cv::Mat sceneImage;
	cv::Mat imageDiff;
	cv::Mat eyeImage_opened;

	double x_min = 150;
	double x_max = 500;
	double y_min = 80;
	double y_max = 300;

	int N_prior_meas = 10;

	cv::Mat glint_kernel;
	cv::Mat glint_element;
	cv::Mat pupil_kernel;
	cv::Mat pupil_element;

	std::vector<cv::Point2d> glintPoints;
	std::vector<cv::Point2d> glintPoints_ordered;
	std::vector<cv::Point2d> glintPoints_prev;

	//excluded for gsltest	
	gt::Cornea *cornea;
	std::vector<double> guesses;
	double error;
	Eigen::Vector3d eigCenter;
	Eigen::Vector3d eigGlint;
	cv::Point3d glintPoint3d;
	Eigen::Vector3d eigLED;

	Camera eyeCam;

	cv::Point2d* pupilEllipsePoints;
	cv::Point2d pupilEllipsePoints_prev[4];

	double gaze_dist;  // The distance between cornea center and gaze point, in meters
};

#endif