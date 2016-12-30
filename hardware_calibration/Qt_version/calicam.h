
#ifndef CALICAM_H
#define CALICAM_H

#include <QWidget>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include "autograb.h"


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

class caliCam : public QWidget
{
    Q_OBJECT
public:
    explicit caliCam(QWidget *parent = 0);
    ~caliCam();
    void setCamDisplaySize(int w, int h);
    void hideCalibButtons();
signals:
    
public slots:

private slots:
    void calibrate(); // call cv::calibrateCamera
    void saveImages();
    void saveParameters();
    void openCam();
    void closeCam();
    void camTimerUp();
    void noSourceSelected();
    void grabCamImg2ListWidget();
    void removeImgFromListWidget();
    void readImgFromFiles();
    void cleanImageList();

private:


  QPushButton *createButton(const QString &text, QWidget *receiver,
                            const char *member);


  // QPixmap camLabelPixmap;
  QLabel * imgSizeLabel; // show how big cam image and display image are
  QLabel * displayLabel;

  QGroupBox *optionGroupBox;
  QRadioButton *fromFileRadioButton;
  QRadioButton *fromCamRadioButton;
  QRadioButton *noSourceRadioButton;




  QListWidget * imageListWidget;
  QLabel * noiLabel; // noi: number of images caputred




  QPushButton * calibPushButton;
  QPushButton * saveImagesPushButton;
  QPushButton * saveParametersPushButton;
  QPushButton * closeCamPushButton;
  QPushButton * grabCamImagePushButton;
  QPushButton * removeImgPushButton;
  QPushButton * removeBadImagesPushButton;

  QVBoxLayout * buttonsVLayout;
  QHBoxLayout * buttonsHLayout1;
  QHBoxLayout * buttonsHLayout2;
  QHBoxLayout * buttonsHLayout3;

  QVBoxLayout * mainLayout;
  QHBoxLayout * optionLayout;
  //QHBoxLayout * buttonsLayout;

  QTimer * camTimer;

  unsigned int cvCamId;
  cv::VideoCapture cvVc;



  std::vector<cv::Mat> cvImageVector;
  std::vector<std::vector<cv::Point2f> >cvCcsVector; // circle centers of circle grid // floating point required, check opencv source code

  std::vector< bool> patternFoundVector;

  bool needSave;

  cv::Size cvBoardSize;

  bool patternFound;
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  std::vector< cv::Mat> rvecs, tvecs;



  bool flipImageBool;

  AutoGrab * ag;
    bool useAutoGrab;

    bool camAreOpen;
};

#endif // CALICAM_H


/*

a image list keep images from camera - a std::vecotr of cv::Mat used to keep those images, synced with the list


*/
