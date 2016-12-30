#include <QtGui>
#include "widget.h"

#include "pages.h"

#include "calidatamethod.h"

#include "iostream"
#include "fstream"

#include <limits>       // std::numeric_limits

Widget::Widget()
{
    iconListWidget = new QListWidget;
    iconListWidget->setViewMode(QListView::IconMode);
    iconListWidget->setIconSize(QSize(96,84));
    iconListWidget->setMovement(QListView::Static);
    iconListWidget->setMaximumWidth(128);
    iconListWidget->setSpacing(12);
    iconListWidget->setMinimumHeight(600);

    eyeCamPage = new EyeCamPage;
    sceneCamPage = new SceneCamPage;
    rigPage = new RigPage;
    captureImagePairsPage = new CaptureImagePairsPage;
    caliPage = new CaliPage;

    pageStackedWidget = new QStackedWidget;
    pageStackedWidget->addWidget(eyeCamPage);
    pageStackedWidget->addWidget(sceneCamPage);
    pageStackedWidget->addWidget(rigPage);
    pageStackedWidget->addWidget(captureImagePairsPage);
    pageStackedWidget->addWidget(caliPage);

    QPushButton *closeButton = new QPushButton(tr("close"));

    createIcons();
    iconListWidget->setCurrentRow(0);


    QHBoxLayout *horizontalLayout = new QHBoxLayout;
    horizontalLayout->addWidget(iconListWidget);
    horizontalLayout->addWidget(pageStackedWidget,1);

    QHBoxLayout * buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(closeButton);

    QVBoxLayout * mainLayout = new QVBoxLayout;
    mainLayout->addLayout(horizontalLayout );
    mainLayout->addStretch(1);
    mainLayout->addSpacing(12);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);

    setWindowTitle(tr("eye-scene cam calibration"));


    // connection
    connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));
    connect(caliPage->calibPushButton,SIGNAL(clicked()),
            this,SLOT(calibrate()));

    showMaximized();

}



void Widget::createIcons(){
    QListWidgetItem * eyeCamButton = new QListWidgetItem(iconListWidget);
    eyeCamButton->setIcon(QIcon(":/images/eyeCam.jpg"));
    eyeCamButton->setText(tr("eye camera\nparameters"));
    eyeCamButton->setTextAlignment(Qt::AlignHCenter);
    eyeCamButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QListWidgetItem * sceneCamButton = new QListWidgetItem(iconListWidget);
    sceneCamButton->setIcon(QIcon(":/images/sceneCam.jpg"));
    sceneCamButton->setText(tr("scene camera\nparameters"));
    sceneCamButton->setTextAlignment(Qt::AlignHCenter);
    sceneCamButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QListWidgetItem * rigButton = new QListWidgetItem(iconListWidget);
    rigButton->setIcon(QIcon(":/images/rig.png"));
    rigButton->setText(tr("rig parameters"));
    rigButton->setTextAlignment(Qt::AlignHCenter);
    rigButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QListWidgetItem * captureImagePairsButton = new QListWidgetItem(iconListWidget);
    //captureImagePairsButton->setIcon(QIcon(":/images/camShoot.png"));
    captureImagePairsButton->setIcon(QIcon(":/images/Two-cameras.jpg"));
    captureImagePairsButton->setText(tr("image pairs"));
    captureImagePairsButton->setTextAlignment(Qt::AlignHCenter);
    captureImagePairsButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QListWidgetItem * caliButton = new QListWidgetItem(iconListWidget);
    caliButton->setIcon(QIcon(":/images/calibration.jpg"));
    caliButton->setText(tr("calibration"));
    caliButton->setTextAlignment(Qt::AlignHCenter);
    caliButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    connect(iconListWidget, SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
            this,SLOT(changePage(QListWidgetItem*,QListWidgetItem*)));
}


void Widget::changePage(QListWidgetItem * current, QListWidgetItem *previous){

    if (!current) current = previous;
    pageStackedWidget->setCurrentIndex(iconListWidget->row(current));

    if (iconListWidget->row(current)==4) caliPageActivating();

}


void Widget::caliPageActivating(){
    //    qDebug()<<"current page: caliPage";

    // fill contents on caliPage

    if (eyeCamPage->getCameraMatrixLoadedFromFile())
        caliPage->printEyeCamIntrinsic(eyeCamPage->getIntrinsicMatrix());


    if (eyeCamPage->getDistCoeffsLoadedFromFile())
        caliPage->printEyeCamDistortion(eyeCamPage->getDistortionVector());


    if (sceneCamPage->getCameraMatrixLoadedFromFile())
        caliPage->printSceneCamIntrinsic(sceneCamPage->getIntrinsicMatrix());


    if (sceneCamPage->getDistCoeffsLoadedFromFile())
        caliPage->printSceneCamDistortion(sceneCamPage->getDistortionVector());

    caliPage->printRigMatrix(rigPage->getRigMatrix());

    caliPage->printScalingFactor(rigPage->getScalingFactor());

    caliPage->printNoip(captureImagePairsPage->getNoip());

}


void Widget::calibrate(){

    qDebug()<<"entering calibrate()";

    CaliDataMethod cdm;

    // check if all needed info valid

    if (captureImagePairsPage->getNoip()){
        cdm.setNumOfImagePairs(captureImagePairsPage->getNoip());
    } else{
        QMessageBox::information(this,
                                 "data incomplete",
                                 "no image pairs of virtual eye cam and scene cam");
        return;
    }


    qDebug()<<"noip set";

    if (eyeCamPage->getCameraMatrixLoadedFromFile())
        cdm.setEyeCameraMatrix(eyeCamPage->getIntrinsicMatrix());
    else {
        QMessageBox::information(this,
                                 "data incomplete",
                                 "eye cam matrix, not available");
        return;
    }


    qDebug()<<"eye cam matrix set";

    if (eyeCamPage->getDistCoeffsLoadedFromFile())
        cdm.setEyeCameradistVector(eyeCamPage->getDistortionVector());
    else {
        QMessageBox::information(this,
                                 "data incomplete",
                                 "eye cam distortion, not available");
        return;
    }

    qDebug()<<"eye cam dist set";


    if (sceneCamPage->getCameraMatrixLoadedFromFile())
        cdm.setSceneCameraMatrix(sceneCamPage->getIntrinsicMatrix());
    else {
        QMessageBox::information(this,
                                 "data incomplete",
                                 "scene cam matrix, not available");
        return;
    }

    qDebug()<<"scene cam matrix set";

    if (sceneCamPage->getDistCoeffsLoadedFromFile())
        cdm.setSceneCameraDistVector(sceneCamPage->getDistortionVector());
    else {
        QMessageBox::information(this,
                                 "data incomplete",
                                 "scene cam distortion, not available");
        return;
    }


    qDebug()<<"scene cam dist set";

    cdm.setRigMatrix(rigPage->getRigMatrix());
    qDebug()<<"rig matrix set";

    cdm.setSf(rigPage->getScalingFactor());
    qDebug()<<"sf set";

    cdm.setCvEyeCcsVector(captureImagePairsPage->getCvEyeCcsVector());
    qDebug()<<"cvEyeCcsVector set";

    cdm.setCvSceneCcsVector(captureImagePairsPage->getCvSceneCcsVector());
    qDebug()<<"cvSceneCcsVector seet";

    cdm.calibrate();
    qDebug()<<"calibrate() called";

    caliPage->setCvM44VirtualEyeCam2SceneCam(cdm.getCvM44VirtualEyeCam2SceneCam());
    qDebug()<<"cvM44VirtualEyeCam2SceneCam set";

    QMessageBox::information(this,
                             "save cali result",
                             "next, select a directory to save calibration result");

    saveResult(cdm.getCvM44VirtualEyeCam2SceneCam());
    qDebug()<<"cali result saved";

    /*
    QMessageBox::information(this,
                             "opengl simulation",
                             "next, call the opengl simulator to show result, use key: z c t u < x q e and arrows to nevigate.");
    system("/home/chi/b/work/showCaliResult/sim /home/chi/b/work/caliProtoshop_Qt/rig_image/caliResult/cvM44SceneCam2VirtualEyeCam.yaml");
*/
    qDebug()<<"leaving calibrate()";
}





void Widget::saveResult(const cv::Mat & x){

    // check

    Q_ASSERT(x.size() == cv::Size(4,4));
    Q_ASSERT(x.type()==CV_64FC1);


    // get directory
    QFileDialog::Options options = QFileDialog::ShowDirsOnly;

    QString directory = QFileDialog::getExistingDirectory(
                this, "select directory to save calibration result",
                "../",options
                );
    if (directory.isEmpty()) return;

    QString qStringTmp;


    // save: from virtual eye cam to scene cam

    qStringTmp = directory + "/cvM44VirtualEyeCam2SceneCam.txt";
    std::string fileName = qStringTmp.toStdString();
    std::ofstream cvM44VirtualEyeCam2SceneCamFile (fileName.c_str());

    std::stringstream ss;
    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);

    if (cvM44VirtualEyeCam2SceneCamFile.is_open()){
        for (int i=0; i<4; i++){
            ss.clear();
            ss.str("");
            for (int j=0; j<4; j++){
                ss<<x.at<double>(i,j)<<'\t';
            }
            cvM44VirtualEyeCam2SceneCamFile<<ss.str()<<'\n';
        }
        cvM44VirtualEyeCam2SceneCamFile.close();
    }

    // save it again using opencv

    qStringTmp = directory +"/cvM44VirtualEyeCam2SceneCam.yaml";
    fileName = qStringTmp.toStdString();
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    fs<<"cvM44VirtualEyeCam2SceneCam"<<x;
    fs.release();




    // save, from scene cam to eye virtual cam

    // invert first
    cv::Mat cvM44SceneCam2VirtualEyeCam = cv::Mat::ones(4,4,CV_64FC1);

    cv::invert(x,cvM44SceneCam2VirtualEyeCam);

    // now, save

    qStringTmp = directory + "/cvM44SceneCam2VirtualEyeCam.txt";
    fileName = qStringTmp.toStdString();
    std::ofstream cvM44SceneCam2VirtualEyeCamFile (fileName.c_str());



    if (cvM44SceneCam2VirtualEyeCamFile.is_open()){
        for (int i=0; i<4; i++){
            ss.clear();
            ss.str("");
            for (int j=0; j<4; j++){
                ss<<cvM44SceneCam2VirtualEyeCam.at<double>(i,j)<<'\t';
            }
            cvM44SceneCam2VirtualEyeCamFile<<ss.str()<<'\n';
        }
        cvM44SceneCam2VirtualEyeCamFile.close();
    }

    // save it again using opencv

    qStringTmp = directory +"/cvM44SceneCam2VirtualEyeCam.yaml";
    fileName = qStringTmp.toStdString();
    fs.open(fileName, cv::FileStorage::WRITE);
    fs<<"cvM44SceneCam2VirtualEyeCam"<<cvM44SceneCam2VirtualEyeCam;
    fs.release();

    // 3d demo of result

    if(
            QMessageBox::Yes ==
            QMessageBox::question(
                this,
                "opengl simulation",
                "use opengl simulator to show the camera pose in 3D?",
                QMessageBox::Yes|QMessageBox::No)
            ){


        // select opengl simulator app

        QFileDialog::Options options = QFileDialog::DontResolveSymlinks;
        QString selectedFilter;
        QString openglAppName =
                QFileDialog::getOpenFileName(
                    this,
                    tr("select opengl simulator"),
                    "..",
                    tr("simulator app (sim)"),
                    &selectedFilter,
                    options
                    );

        QString caliResultFileName =
                QFileDialog::getOpenFileName(
                    this,
                    tr("select calibration result file"),
                    "..",
                    tr("calibration result file (cvM44SceneCam2VirtualEyeCam.yaml)"),
                    &selectedFilter,
                    options
                    );
        QMessageBox::information(
                    this,
                    "how to use the simulator",
                    "use following keys to nevigate: z c t u < x q e and arrow keys"
                    );

        system((openglAppName + " " + caliResultFileName).toStdString().c_str());




    }

}











