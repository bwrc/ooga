#ifndef AUTOGRABTWOCAM_H
#define AUTOGRABTWOCAM_H




#include <QtGui>

#include <opencv2/opencv.hpp>

#define N 34
// N: normal frame rate, 33 ms, N frames over 1 second

#define TIMESPAN 100
// 1000 msecons = 1 second

#define THRESHOLD 2

#define INDEX_BOUND_THRESHOLD 3

QT_BEGIN_NAMESPACE
class QTime;
QT_END_NAMESPACE


class AutoGrabTwoCam
{
public:
    AutoGrabTwoCam();


    void ini();

    bool pushAndCheck(
            bool eyePatternFoundFlag,
            bool scenePatternFoundFlag,
            double eyePointsCenterX,
            double scenePointsCenterX,
            double eyePointsCenterY,
            double scenePointsCenterY);

    void writeInfoOnImage(cv::Mat & eyeImg, cv::Mat & sceneImg);

private:

    // main data
    int timeStamps[N];
    double eyePointsCentersX[N];
    double scenePointsCentersX[N];
    double eyePointsCentersY[N];
    double scenePointsCentersY[N];
    bool eyePatternFoundFlags[N];
    bool scenePatternFoundFlags[N];

    // index
    unsigned int indexBound,  frameRate;

    // counter
    unsigned int frameCount,  goodImagePairCount;

    // varialbes, computing
    double eye_mx, eye_my; // mean of x and y
    double eye_ssx, eye_ssy; // some of saures, x and y
    double eye_stdx, eye_stdy; // standard deviation, x and y
    double scene_mx, scene_my; // mean of x and y
    double scene_ssx, scene_ssy; // some of saures, x and y
    double scene_stdx, scene_stdy; // standard deviation, x and y

    // time facitliy
    QTime myTime;



    // method
    void shift2left(
            //int timeElapsed,
            bool eyePatternFoundFlag,
            bool scenePatternFoundFlag,
            double eyePointsCenterX,
            double scenePointsCenterX,
            double eyePointsCenterY,
            double scenePointsCenterY
            );

    void clearTimeStamps();


};

#endif // AUTOGRABTWOCAM_H
