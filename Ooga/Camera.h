#ifndef CAMERA_H
#define CAMERA_H


/*
 *	This class uses the OpenCV right-handed coordinate system:
 *		x+ : right
 *		x- : left
 *		y+ : down
 *		y- : up
 *		z+ : towards monitor
 *		z- : towards viewer
 */


#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/*	
 *	http://en.wikipedia.org/wiki/Angle_of_view
 *
 *	double fy = h_2 / tan(DEGTORAD(FOV_y) / 2.0);
 *	double fx = fy;
 */

class Camera {

public:

    Camera();

    /* Copy contructor */
    Camera(const Camera &other) {

        intrinsic_matrix = other.intrinsic_matrix.clone();
        distortion       = other.distortion.clone();

    }


    /* Assignment operator */
    Camera & operator= (const Camera &other) {

        // protect against invalid self-assignment
        if(this != &other) {

            intrinsic_matrix = other.intrinsic_matrix.clone();
            distortion       = other.distortion.clone();

        }

        // by convention, always return *this
        return *this;

    }



    /*
     * The matrix is to be given in the column-major order:
     * The first three elements should be the first column and the following
     * three elements the second colum and the last three should make for
     * the last colum.
     */
    void setIntrinsicMatrix(const double intr[9]);
    void setDistortion(const double _dist[5]);

	//overloads for matrices
	void setIntrinsicMatrix(const cv::Mat _intr);
	void setDistortion(const cv::Mat _dist);

    const cv::Mat& getIntrisicMatrix() const {return intrinsic_matrix;}
    const cv::Mat& getDistortion() const {return distortion;}

    void pixToWorld(const std::vector<cv::Point2d> &image_points, std::vector<cv::Point3d> &vec3D) const;
    void pixToWorld(double u, double v, cv::Point3d &p3D) const;
    void worldToPix(const cv::Point3d &p3D, double *u, double *v) const;

private:

    cv::Mat intrinsic_matrix;
    cv::Mat distortion;
};


#endif

