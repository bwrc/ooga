#ifndef SG_COMMON_H
#define SG_COMMON_H

#include <chrono>
#include <vector>
#include <opencv2/core/core.hpp>

typedef std::chrono::high_resolution_clock hrclock;
typedef std::chrono::milliseconds msecs;

struct TGrabStatistics{
	msecs grabTime;
	msecs frameTime;
	msecs waitTime;
};

struct TTrackingResult{
	cv::Point2d pupilCenter2D;
	cv::Point3d pupilCenter3D;
	std::vector<cv::Point2d> pupilEllipsePoints;
	cv::RotatedRect pupilEllipse;

	std::vector<cv::Point2d> glintPoints;
	cv::Point3d corneaCenter3D;
	cv::Point3d gazeDirectionVector;

	double score;

};


#endif