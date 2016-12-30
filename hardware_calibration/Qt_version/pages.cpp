#include <QtGui>

#include "pages.h"
#include "calidatamethod.h"

EyeCamPage::EyeCamPage(QWidget *parent):QWidget(parent)
{


    eyeCam = new SingleCam;


    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(eyeCam);
    setLayout(mainLayout);


}

SceneCamPage::SceneCamPage(QWidget *parent):QWidget(parent)
{
    sceneCam = new SingleCam;

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(sceneCam);
    setLayout(mainLayout);

}

RigPage::RigPage(QWidget *parent):QWidget(parent)
{

    sf = 0.137625838926174;

    QString note = "RIG PARAMETERS ARE HARD CODED NUMBERS, CHANGE SOURCE CODE IF NEEDED";
    note.append("\n\n");
    note.append("d_small : vertical (=horizental) distance between two circle centers on small board");
    note.append("\n\n");
    note.append("d_big : vertical (=horizental) distance between two circle centers on big board");
    note.append("\n\n");
    note.append("scaling factor : sf = d_small / d_big = " + QString::number(sf));
    note.append("\n\n");
    note.append("unit length is d_big = 47.68 mm");
    note.append("\n\n");
    note.append("pattern:");
    note.append("\n\n");
    for (int i = 0; i<5; i++) note.append("o\t\to\t\to\t\to\n\to\t\to\t\to\t\to\n");
    note.append("o\t\to\t\to\t\to");
    note.append("\n\n");
    note.append("(0,0,0) at the upper left circle center, x-axis to right, y-axis down");
    note.append("\n\n");
    note.append("\n4x4 matrix tells how to rotate/translate small board coordiante to the big board coordiante");
    note.append("\ni.e., first 3 element of first column gives the big board x-axis observed in small board o-x-y-z");
    note.append("\nassuming no translation");


    QLabel * noteLabel = new QLabel(note);


    cvM44SmallBord2BigBoard = cv::Mat::eye(cv::Size(4,4),CV_64FC1);
    /*
    1
    0
    0
    0

    0
    -1
    0
    0

    0
    0
    -1
    0

    -6.68590604026846
    5.68817114093960
    -43.20469798657718
    1
*/

    if (1) {  // the "normal" small-board-to-big-board
      cvM44SmallBord2BigBoard.at<double>(0,0) = -1; // this was positive, should've been negative t: Miika
      cvM44SmallBord2BigBoard.at<double>(1,1) = 1;  // this was negative, should've been positive t: Miika
      cvM44SmallBord2BigBoard.at<double>(2,2) = -1; // this was negative
      cvM44SmallBord2BigBoard.at<double>(0,3) = 5.68817114093960;   // this was positive and at (0,3)
      cvM44SmallBord2BigBoard.at<double>(1,3) = 6.68590604026846;   // this was negative and at (1,3)
      cvM44SmallBord2BigBoard.at<double>(2,3) = -43.20469798657718; // this was negative

    } else {  // for computing the left-to-right eye transformation
      cvM44SmallBord2BigBoard.at<double>(0,3) = 24.8; // (in units of 2.5 mm: 62 mm / 2.5 mm) 
    }

    
    QTableWidget * m44TableWidget = new QTableWidget;
    m44TableWidget->setColumnCount(4);
    m44TableWidget->setRowCount(4);
    for (int r = 0; r<4; r++){
        for (int c = 0; c<4; c++){
            QTableWidgetItem * item = new QTableWidgetItem;
            item->setText(QString::number(cvM44SmallBord2BigBoard.at<double>(r,c)));
            item->setFlags(Qt::NoItemFlags);
            m44TableWidget->setItem(r,c,item);
        }
    }


    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(noteLabel);
    mainLayout->addWidget(m44TableWidget);
    setLayout(mainLayout);

}

CaliPage::CaliPage(QWidget *parent):QWidget(parent)
{


    /* layout

        eye cam intrinsic:      a matrix here   comment 0
        eye cam dist:           a vector here   comment 1
        scene cam intrinsic:    a matrix here   comment 2
        scene cam dist:         a vector here   comment 3
        rig para:               a matrix here   comment 4
        scaling:                a double here   comment 5
        eye-scene image pairs   n pairs read            6
        calib button                                    7
        result: (virtual) eye->scene:   a matrix here   8
    */

    // layout objects



      eyeCamIntrinsicLabel = new QLabel("eye cam intrinsic: ");
      eyeCamDstLabel = new QLabel("eye cam distortion: ");
      sceneCamIntrinsicLabel = new QLabel("scene cam intrinsic: ");
      sceneCamDstLabel = new QLabel("scene cam distortion: ");
      rigLabel = new QLabel("rig parementer: ");
      scaleHeaderLabel = new QLabel("scaling factor: ");
      scaleValueLabel = new QLabel("scaling factor value not loaded");
      imagePairsHeaderLabel = new QLabel("(virtual) eye - scene image pairs: ");
      imagePairsContentLabel = new QLabel("image pairs not loaded");
      calibrationResultLabel = new QLabel("calibration result: ");

    calibPushButton = new QPushButton("calibrate now");

    eyeCamIntrinsicTable = new QTableWidget;
    eyeCamDstTable = new QTableWidget;
    sceneCamIntrinsicTable = new QTableWidget;
    sceneCamDstTable = new QTableWidget;
    rigTable = new QTableWidget;
    calibResultTable = new QTableWidget;

    // layout

    QGridLayout *mainLayout = new QGridLayout;


    mainLayout->addWidget(eyeCamIntrinsicLabel,0,0);
    mainLayout->addWidget(eyeCamIntrinsicTable,0,1);

    mainLayout->addWidget(eyeCamDstLabel,1,0);
    mainLayout->addWidget(eyeCamDstTable,1,1);

    mainLayout->addWidget(sceneCamIntrinsicLabel,2,0);
    mainLayout->addWidget(sceneCamIntrinsicTable,2,1);

    mainLayout->addWidget(sceneCamDstLabel,3,0);
    mainLayout->addWidget(sceneCamDstTable,3,1);

    mainLayout->addWidget(rigLabel,4,0);
    mainLayout->addWidget(rigTable,4,1);

    mainLayout->addWidget(scaleHeaderLabel,5,0);
    mainLayout->addWidget(scaleValueLabel,5,1);

    mainLayout->addWidget(imagePairsHeaderLabel,6,0);
    mainLayout->addWidget(imagePairsContentLabel,6,1);

    mainLayout->addWidget(calibPushButton,7,0,1,2);

    mainLayout->addWidget(calibrationResultLabel,8,0);
    mainLayout->addWidget(calibResultTable,8,1);





    setLayout(mainLayout);




    // data ini
    // eye cam
    // scene cam
    // rig
    // scaling



}




CaptureImagePairsPage::CaptureImagePairsPage (QWidget *parent) : QWidget (parent)
{
    twoCam = new TwoCam;


    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(twoCam);
    setLayout(mainLayout);

}


CaptureImagePairsPage::~CaptureImagePairsPage(){
    delete twoCam;
}


void CaliPage:: printEyeCamIntrinsic(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(3,3));
    Q_ASSERT(x.type()==CV_64FC1);



    // write table
    eyeCamIntrinsicTable->clear();
    eyeCamIntrinsicTable->setRowCount(3);
    eyeCamIntrinsicTable->setColumnCount(3);
    for (int row = 0; row <3; row ++){
        for (int col = 0; col<3; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            eyeCamIntrinsicTable->setItem(row,col,item);
        }
    }




}

void CaliPage:: printEyeCamDistortion(const cv::Mat & x){

    Q_ASSERT(x.size()==cv::Size(1,5)); // .size() returns cols x rows!
    Q_ASSERT(x.type()==CV_64FC1);


    // write table
    eyeCamDstTable->clear();
    eyeCamDstTable->setRowCount(1);
    eyeCamDstTable->setColumnCount(5);
    for (int row = 0; row <5; row ++){
        for (int col = 0; col<1; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            eyeCamDstTable->setItem(0,row,item);
        }
    }

}

void CaliPage:: printSceneCamIntrinsic(const cv::Mat & x){

    Q_ASSERT(x.size()==cv::Size(3,3));
    Q_ASSERT(x.type()==CV_64FC1);

    // write table
    sceneCamIntrinsicTable->clear();
    sceneCamIntrinsicTable->setRowCount(3);
    sceneCamIntrinsicTable->setColumnCount(3);
    for (int row = 0; row <3; row ++){
        for (int col = 0; col<3; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            sceneCamIntrinsicTable->setItem(row,col,item);
        }
    }





}

void CaliPage:: printSceneCamDistortion(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(1,5));
    Q_ASSERT(x.type()==CV_64FC1);

    // write table
    sceneCamDstTable->clear();
    sceneCamDstTable->setRowCount(1);
    sceneCamDstTable->setColumnCount(5);
    for (int row = 0; row <5; row ++){
        for (int col = 0; col<1; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            sceneCamDstTable->setItem(0,row,item);
        }
    }


}

void CaliPage:: printRigMatrix(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(4,4));
    Q_ASSERT(x.type()==CV_64FC1);


    // write table
    rigTable->clear();
    rigTable->setRowCount(4);
    rigTable->setColumnCount(4);
    for (int row = 0; row <4; row ++){
        for (int col = 0; col<4; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            rigTable->setItem(row,col,item);
        }
    }

}


void CaliPage:: printCvM44VirtualEyeCam2SceneCam(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(4,4));
    Q_ASSERT(x.type()==CV_64FC1);


    // write table
    calibResultTable->clear();
    calibResultTable->setRowCount(4);
    calibResultTable->setColumnCount(4);
    for (int row = 0; row <4; row ++){
        for (int col = 0; col<4; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(x.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            calibResultTable->setItem(row,col,item);
        }
    }

}








void CaliPage:: printScalingFactor(double x){
    scaleValueLabel->setText(QString::number(x));

}

void CaliPage:: printNoip(unsigned int x){

    imagePairsContentLabel->setText(
                QString::number(x)
                +
                " (make sure all images have patterns! there is a button to remove bad images)"
                );
} // noip: number of image pairs

