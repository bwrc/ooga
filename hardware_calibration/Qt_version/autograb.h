#ifndef AUTOGRAB_H
#define AUTOGRAB_H

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


class AutoGrab
{
public:
    AutoGrab();

    void ini();

    bool pushAndCheck(
            //            int timeElapsed,
            bool patternFoundFlag,
            double pointsCenterX, double pointsCenterY); // push new image into class

    void writeInfoOnImage(cv::Mat & img);

private:

    // main data
    int timeStamps[N];
    double pointsCentersX[N], pointsCentersY[N];
    bool patternFoundFlags[N];

    // index
    unsigned int indexBound, frameRate;

    // counter
    unsigned int frameCount, goodImageCount;

    // varialbes, computing
    double mx, my; // mean of x and y
    double ssx, ssy; // some of saures, x and y
    double stdx, stdy; // standard deviation, x and y

    // time facitliy
    QTime myTime;



    // method
    void shift2left(
            //int timeElapsed,
            bool patternFoundFlag, double pointsCenterX, double pointsCenterY);
    void clearTimeStamps();
};

#endif // AUTOGRAB_H
