#ifndef TWOCAM_H
#define TWOCAM_H

#include <QWidget>
#include <opencv2/opencv.hpp>

#include "autograbtwocam.h"

QT_BEGIN_NAMESPACE
class QCheckBox;
class QGroupBox;
class QHBoxLayout;
class QImage;
class QLabel;
class QListWidget;
class QListWidgetItem;
class QPushButton;
class QRadioButton;
class QTimer;
class QTime;
class QVBoxLayout;
QT_END_NAMESPACE

class TwoCam : public QWidget
{
    Q_OBJECT
public:
    explicit TwoCam(QWidget *parent = 0);
    ~TwoCam();

    // getters
    unsigned int getNoip();
    const std::vector< std::vector <cv::Point2f> > & getCvEyeCcsVector(){
        return cvEyeCcsVector;
    }
    const std::vector< std::vector <cv::Point2f > > & getCvSceneCcsVector(){
        return  cvSceneCcsVector;
    }


signals:
    
public slots:



private slots:
    void saveImages();
    void openCam();
    void closeCam();
    void camTimerUp();
    void noSourceSelected();
    void grabCamImage2ListWidget();
    void removeEyeImageFromListWidget();
    void removeSceneImageFromListWidget();
    void readImageFiles();
    void cleanImageList();


private:


    QPushButton *createButton(const QString &text, QWidget *receiver, const char *member);



    QLabel * eyeTopLabel;
    QLabel * sceneTopLabel;

    QLabel * eyeDisplayLabel;
    QLabel * sceneDisplayLabel;

    QLabel * imgSizeLabel; // show how big cam image and display image are
    QLabel * noiLabel; // noi: number of images caputred

    QListWidget * eyeImageListWidget;
    QListWidget * sceneImageListWidget;


    QGroupBox *optionGroupBox; // from files, from camera, nothing

    QRadioButton *fromFileRadioButton;
    QRadioButton *fromCamRadioButton;
    QRadioButton *noSourceRadioButton;

    QPushButton * saveImagesPushButton;
    QPushButton * closeCamPushButton;
    QPushButton * grabCamImagePushButton;
    QPushButton * removeEyeImgPushButton;
    QPushButton * removeSceneImgPushButton;
    QPushButton * removeBadImagesPushButton;


    QTimer * camTimer;


    QHBoxLayout * buttonsHLayout;
    QVBoxLayout * mainLayout;
    QHBoxLayout * optionLayout;


    unsigned int eyeCamId;
    unsigned int sceneCamId;

    cv::VideoCapture cvEyeVc;
    cv::VideoCapture cvSceneVc;
/*
    cv::Mat cvEyeCamImage;
    cv::Mat cvSceneCamImage;
    cv::Mat cvEyeFileImage;
    cv::Mat cvSceneFileImage;
    */
    std::vector<cv::Mat> cvEyeImageVector;
    std::vector<cv::Mat> cvSceneImageVector;
    std::vector<cv::Point2f> cvEyeCcs; // eye image, circle centers in one image
    std::vector<cv::Point2f> cvSceneCcs; // scene image, circle centers in one image
    std::vector<std:: vector< cv::Point2f> > cvEyeCcsVector; // a vec of cvEyeCcs
    std::vector<std:: vector< cv::Point2f> > cvSceneCcsVector; // a vec of cvSceneCcs
    std::vector< bool> eyePatternFoundVector;
    std::vector< bool> scenePatternFoundVector;

    bool needSave;

    cv::Size cvBoardSize;
    bool eyePatternFound;
    bool scenePatternFound;



    bool flipImageBool;

    bool useAutoGrab;
    AutoGrabTwoCam * ag;

    bool camAreOpen;

};

#endif // TWOCAM_H
