#include "Camera.h"
#include <math.h>
#include <vector>


Camera::Camera() {

	this->intrinsic_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
	this->distortion = cv::Mat::zeros(1, 5, CV_64FC1);
}


/*
 * The matrix is to be given in the column-major order:
 * The first three elements should be the first column and the following
 * three elements the second colum and the last three should make for
 * the last colum.
 */
void Camera::setIntrinsicMatrix(const double intr[9]) {

	intrinsic_matrix.at<double>(0, 0) = intr[0];
	intrinsic_matrix.at<double>(1, 0) = intr[1];
	intrinsic_matrix.at<double>(2, 0) = intr[2];
	intrinsic_matrix.at<double>(0, 1) = intr[3];
	intrinsic_matrix.at<double>(1, 1) = intr[4];
	intrinsic_matrix.at<double>(2, 1) = intr[5];
	intrinsic_matrix.at<double>(0, 2) = intr[6];
	intrinsic_matrix.at<double>(1, 2) = intr[7];
	intrinsic_matrix.at<double>(2, 2) = intr[8];

}


void Camera::setDistortion(const double _dist[5]) {

	distortion.at<double>(0) = _dist[0];
	distortion.at<double>(1) = _dist[1];
	distortion.at<double>(2) = _dist[2];
	distortion.at<double>(3) = _dist[3];
	distortion.at<double>(4) = _dist[4];

}


/*
 *	It is important to notice that u and v are in a 2D coordinate system with the origin at the upper
 *	left corner of the image.
 */
void Camera::pixToWorld(const std::vector<cv::Point2d> &image_points, std::vector<cv::Point3d> &vec3D) const {

	if(vec3D.size() == 0) {
		vec3D.resize(image_points.size());
	}


	const double cx = intrinsic_matrix.at<double>(0, 2);
	const double cy = intrinsic_matrix.at<double>(1, 2);
	const double fx = intrinsic_matrix.at<double>(0, 0);
	const double fy = intrinsic_matrix.at<double>(1, 1);


	std::vector<cv::Point2d> dstv(image_points.size());

	// gives u' and v'
	cv::undistortPoints(image_points,		// input points
						dstv,				// output points
						intrinsic_matrix,	// intrinsic camera matrix
						distortion,			// distortion coefficients
						cv::Mat(),			// rectification transformation in the object space, unused
						intrinsic_matrix);	// use this matrix to place u' and v' to the same coord system as u and v


	size_t sz = image_points.size();

	for(size_t i = 0; i < sz; ++i) {

		const double uc = dstv[i].x;
		const double vc = dstv[i].y;

		const double x = (uc - cx) / fx;
		const double y = (vc - cy) / fy;


		/**************************************************************************
		 * Get the results
		 * x' = x/z
		 * y' = y/z
		 * we can select z arbitrarily. We set it to 1 ==> x = x', y = y' and z = 1
		 ***************************************************************************/

		// normalise
		const double len = std::sqrt(x*x + y*y + 1.0);

		cv::Point3d &curPoint = vec3D[i];
		curPoint.x = x / len;
		curPoint.y = y / len;
		curPoint.z = 1. / len;

	}

}


/*
 *	It is important to notice that u and v are in a 2D coordinate system with the origin at the upper
 *	left corner of the image.
 */
void Camera::pixToWorld(double u, double v, cv::Point3d &p3D) const {

	std::vector<cv::Point2d> image_point(1);
	image_point[0].x = u;
	image_point[0].y = v;

	std::vector<cv::Point3d> object_point(1);
	pixToWorld(image_point, object_point);

	p3D = object_point[0];
}


/*
 *	http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html
 *	It is important to notice that u and v are in a 2D coordinate system with the origin at the upper
 *	left corner of the image.
 */
void Camera::worldToPix(const cv::Point3d &p3D, double *u, double *v) const {

	// A list containing a single point
	std::vector<cv::Point3f> pointList;

	pointList.push_back(cv::Point3f(p3D.x, p3D.y, p3D.z));

	// rvec = tvec = <0.0, 0.0, 0.0>
	cv::Mat rtvec = cv::Mat::zeros(3, 1, CV_64F);

	// Get projection using opencv
	std::vector<cv::Point2f> projectedPointsList;
	cv::projectPoints(pointList, rtvec, rtvec, this->intrinsic_matrix, this->distortion, projectedPointsList);
	const cv::Point2f &displayPoint = projectedPointsList[0];

	// Write the result
	*u = displayPoint.x;
	*v = displayPoint.y;
}

