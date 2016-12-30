#ifndef PAGES_H
#define PAGES_H

#include <QWidget>
#include <QTableWidget>

#include "singlecam.h"
#include "twocam.h"

class EyeCamPage : public QWidget
{

public:
    EyeCamPage(QWidget *parent = 0);

    const cv::Mat & getIntrinsicMatrix(){
        return eyeCam->getIntrinsicMatrix();
    }

    const cv::Mat & getDistortionVector() {
        return eyeCam->getDistortionVector();
    }

    bool getCameraMatrixLoadedFromFile(){
        return eyeCam->getCameraMatrixLoadedFromFile();
    }

    bool         getDistCoeffsLoadedFromFile(){
        return eyeCam->getDistCoeffsLoadedFromFile();
    }

private:
    SingleCam * eyeCam;
};

class SceneCamPage : public QWidget
{

public:
    SceneCamPage(QWidget * parent = 0);


    const cv::Mat & getIntrinsicMatrix(){
        return sceneCam->getIntrinsicMatrix();
    }

    const cv::Mat & getDistortionVector() {
        return sceneCam->getDistortionVector();
    }

    bool getCameraMatrixLoadedFromFile(){
        return sceneCam->getCameraMatrixLoadedFromFile();
    }

    bool         getDistCoeffsLoadedFromFile(){
        return sceneCam->getDistCoeffsLoadedFromFile();
    }

private:
    SingleCam * sceneCam;
};

class RigPage : public QWidget
{

public:
    RigPage(QWidget * parent = 0);
    double getScalingFactor (){return sf;}
    const cv::Mat & getRigMatrix(){return cvM44SmallBord2BigBoard;}

private:
    double sf; // scaling factor
    cv::Mat cvM44SmallBord2BigBoard; // how to move VIRTUAL eye o-x-y-z to scene o-x-y-z
};

class CaliPage : public QWidget
{

public:
    void setCvM44VirtualEyeCam2SceneCam(const cv::Mat & x){
        cvM44VirtualEyeCam2SceneCam = x.clone();
        printCvM44VirtualEyeCam2SceneCam(cvM44VirtualEyeCam2SceneCam);
    }

public slots:



    void printEyeCamIntrinsic(const cv::Mat & x);
    void printEyeCamDistortion(const cv::Mat & x);
    void printSceneCamIntrinsic(const cv::Mat & x);
    void printSceneCamDistortion(const cv::Mat & x);
    void printRigMatrix(const cv::Mat & x);
    void printScalingFactor(double x);
    void printNoip(unsigned int x); // noip: number of image pairs

    void printCvM44VirtualEyeCam2SceneCam(const cv::Mat & x);

public:
    CaliPage(QWidget *parent = 0);
QPushButton * calibPushButton;

private:


    QLabel * eyeCamIntrinsicLabel;
    QLabel * eyeCamDstLabel;
    QLabel * sceneCamIntrinsicLabel;
    QLabel * sceneCamDstLabel;
    QLabel * rigLabel;
    QLabel * scaleHeaderLabel;
    QLabel * scaleValueLabel;
    QLabel * imagePairsHeaderLabel;
    QLabel * imagePairsContentLabel;
    QLabel * calibrationResultLabel;




    QTableWidget * eyeCamIntrinsicTable;
    QTableWidget * eyeCamDstTable;
    QTableWidget * sceneCamIntrinsicTable;
    QTableWidget * sceneCamDstTable;
    QTableWidget * rigTable;
    QTableWidget * calibResultTable;


    // data
    cv::Mat cvM44VirtualEyeCam2SceneCam;
};



class CaptureImagePairsPage : public QWidget
{
public:
    CaptureImagePairsPage(QWidget *parent = 0);
    ~CaptureImagePairsPage();
    unsigned int getNoip(){
        return twoCam->getNoip();
    }


    const std::vector< std::vector <cv::Point2f> > & getCvEyeCcsVector(){
        return twoCam->getCvEyeCcsVector();
    }

    const std::vector< std::vector <cv::Point2f> > & getCvSceneCcsVector(){
        return twoCam->getCvSceneCcsVector();
    }

private:
    TwoCam * twoCam;
};


#endif // PAGES_H
