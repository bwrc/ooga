#ifndef SINGLECAM_H
#define SINGLECAM_H


#include<QDialog>


#include "calicam.h"

QT_BEGIN_NAMESPACE
class QDialogButtonBox;
class QFileInfo;
class QTabWidget;
class QTableWidget;
QT_END_NAMESPACE



class UseFile : public QWidget
{
    Q_OBJECT
public:
    UseFile( QWidget * parent = 0);

    const cv::Mat & getIntrinsicMatrix();
    const cv::Mat & getDistortionVector();
    bool  getCameraMatrixLoadedFromFile() {return cameraMatrixLoadedFromFile;}
    bool  getDistCoeffsLoadedFromFile() {return distCoeffsLoadedFromFile;}

public slots:
    void loadCamFile(); // camera matrix, the intrinsic parametner
    void loadDstFile(); // distoration parameter

private:
    QLabel * camLabel;
    QLabel * dstLabel;

    QPushButton * loadCamFilePushButton;
    QPushButton * loadDstFilePushButton;

    QTableWidget * camTableWidget;
    QTableWidget * dstTableWidget;

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;


    bool cameraMatrixLoadedFromFile;
    bool distCoeffsLoadedFromFile;
};


class UseCam : public QWidget
{
    Q_OBJECT
public:
    UseCam(QWidget * parent = 0);
    ~UseCam();
    caliCam * caliSingleCam;
};


class SingleCam : public QDialog
{
    Q_OBJECT

public:
    SingleCam( QWidget *parent = 0);

    const cv::Mat & getIntrinsicMatrix(){
        return useFile->getIntrinsicMatrix();
    }

    const cv::Mat & getDistortionVector() {
        return useFile->getDistortionVector();
    }

    bool  getCameraMatrixLoadedFromFile(){
        return useFile->getCameraMatrixLoadedFromFile();
    }

    bool  getDistCoeffsLoadedFromFile(){
        return useFile->getDistCoeffsLoadedFromFile();
    }

private:
    QTabWidget *tabWidget;

    UseFile * useFile;
    UseCam * useCam;




};


#endif // SINGLECAM_H
