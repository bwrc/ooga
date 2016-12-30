#ifndef CALIDATAMETHOD_H
#define CALIDATAMETHOD_H

#include <opencv2/opencv.hpp>



class CaliDataMethod
{
public:

    // functions listed in the order they are to be called !!!!!!!!!

    // setters
    void setNumOfImagePairs(unsigned int x);

    void setEyeCameraMatrix(const cv::Mat & x);
    void setEyeCameradistVector(const cv::Mat & x);
    void setSceneCameraMatrix(const cv::Mat & x);
    void setSceneCameraDistVector(const cv::Mat & x);
    void setRigMatrix(const cv::Mat & x);
    void setSf(double x);
    void setCvEyeCcsVector(const std::vector<std::vector<cv::Point2f> > & x);
    void setCvSceneCcsVector(const std::vector<std::vector<cv::Point2f> > & x);

    void calibrate();

    // getters
    const cv::Mat & getCvM44VirtualEyeCam2SceneCam(){
        return cvM44VirtualEyeCam2SceneCam;
    }


    // other functions

    CaliDataMethod();

private:

    unsigned int noip;

    cv::Mat eyeCameraMatrix;
    cv::Mat eyeCameraDistVector;

    cv::Mat sceneCameraMatrix;
    cv::Mat sceneCameraDistVector;

    cv::Mat rigMatrix;

    double sf; // scaling factor

    // image points
    std::vector<cv::Point2f> cvEyeCcs; // eye image, circle centers in one image
    std::vector<cv::Point2f> cvSceneCcs; // scene image, circle centers in one image
    std::vector<std:: vector< cv::Point2f> > cvEyeCcsVector; // a vec of cvEyeCcs
    std::vector<std:: vector< cv::Point2f> > cvSceneCcsVector; // a vec of cvSceneCcs

    // object points
    std::vector<cv::Point3f> cvObjectPointsSmallBoard; // float, check opencv code
    std::vector<cv::Point3f> cvObjectPointsBigBoard; // float, check opencv code





    // other things
    std::vector< cv::Mat > rmats, tvecs;
    cv::Mat rvecVirtualEyeCam2SmallBoard;
    cv::Mat tvecVirtualEyeCam2SmallBoard;
    cv::Mat cvM44VirtualEyeCam2SmallBoard;


    cv::Mat rvecSceneCam2BigBoard;
    cv::Mat tvecSceneCam2BigBoard;
    cv::Mat cvM44SceneCam2BigBoard;

    cv::Mat cvM44BigBoard2SceneCam;

    cv::Mat cvM44VirtualEyeCam2BigBoard;

    cv::Mat cvM44VirtualEyeCam2SceneCam;

    cv::Mat rvecVirtualEyeCam2SceneCam;
    cv::Mat tvecVirtualEyeCam2SceneCam;


    cv::Mat tmpM33;

    void matrixProduct(cv::Mat & a, cv::Mat & b, cv::Mat & c);
};

#endif // CALIDATAMETHOD_H
