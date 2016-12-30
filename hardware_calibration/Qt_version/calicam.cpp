#include "calicam.h"
#include <QtGui>

#include <iostream>
#include <fstream>

#include <limits>       // std::numeric_limits

#include <opencv2/opencv.hpp>

caliCam::~caliCam(){
    if (!cvVc.isOpened())  cvVc.release();
    camTimer->stop();
}

caliCam::caliCam(QWidget *parent) :
    QWidget(parent)
{

    camTimer = new QTimer;



    // objects
    displayLabel = new QLabel;
    displayLabel->setAlignment(Qt::AlignHCenter);
    setCamDisplaySize(640,480);
    displayLabel->setText("no image to show up here for the moment");

    imgSizeLabel = new QLabel;
    imgSizeLabel->clear();

    noiLabel = new QLabel("number of images in aobve list: 0");

    optionGroupBox = new QGroupBox("select image source");

    fromFileRadioButton = new QRadioButton;
    fromFileRadioButton->setText("load image files");
    fromCamRadioButton = new QRadioButton;
    fromCamRadioButton->setText("grab images from cam");
    noSourceRadioButton = new QRadioButton;
    noSourceRadioButton->setText("nothing to select");
    noSourceRadioButton->setChecked(true);

    imageListWidget = new QListWidget;
    imageListWidget->setViewMode(QListView::IconMode);
    imageListWidget->setIconSize(QSize(128,96));
    imageListWidget->setMovement(QListView::Free);
    //imageListWidget->setMinimumWidth(128*7cameraMatrix);
    imageListWidget->setMinimumWidth(128*3);
    imageListWidget->setMinimumHeight(96*1.5);
    imageListWidget->setSpacing(12);
    imageListWidget->setFlow(QListWidget::LeftToRight);

    calibPushButton = createButton("calib now",this,SLOT(calibrate()));
    calibPushButton->setDisabled(true);
    saveImagesPushButton = createButton("save images in list",this,SLOT(saveImages()));
    saveImagesPushButton->setDisabled(true);
    saveParametersPushButton = createButton("save calib parameters",this,SLOT(saveParameters()));
    saveParametersPushButton->setDisabled(true);
    closeCamPushButton = createButton("close cam",this,SLOT(closeCam()));
    closeCamPushButton->setDisabled(true);
    grabCamImagePushButton = createButton("grab cam image",this, SLOT(grabCamImg2ListWidget()));
    grabCamImagePushButton->setDisabled(true);
    removeImgPushButton = createButton("remove current img from list",this,SLOT(removeImgFromListWidget()));
    removeImgPushButton->setDisabled(true);
    removeBadImagesPushButton = createButton("remove bad images",this,SLOT(cleanImageList()));
    removeBadImagesPushButton->setDisabled(true);

    // layouts
    optionLayout = new QHBoxLayout;
    optionLayout->addWidget(fromFileRadioButton);
    optionLayout->addWidget(fromCamRadioButton);
    optionLayout->addWidget(noSourceRadioButton);
    optionGroupBox->setLayout(optionLayout);


    buttonsVLayout = new QVBoxLayout;
    buttonsHLayout1 = new QHBoxLayout;
    buttonsHLayout2 = new QHBoxLayout;
    buttonsHLayout3 = new QHBoxLayout;


    buttonsHLayout1->addWidget(grabCamImagePushButton);
    buttonsHLayout1->addWidget(closeCamPushButton);

    buttonsHLayout2->addWidget(removeImgPushButton);
    buttonsHLayout2->addWidget(removeBadImagesPushButton);
    buttonsHLayout2->addWidget(saveImagesPushButton);

    buttonsHLayout3->addWidget(calibPushButton);
    buttonsHLayout3->addWidget(saveParametersPushButton);

    buttonsVLayout->addLayout(buttonsHLayout1);
    buttonsVLayout->addLayout(buttonsHLayout2);
    buttonsVLayout->addLayout(buttonsHLayout3);

    mainLayout = new QVBoxLayout;

    mainLayout->addWidget(imgSizeLabel);
    mainLayout->addWidget(displayLabel);
    mainLayout->addWidget(imageListWidget);
    mainLayout->addWidget(noiLabel);
    mainLayout->addWidget(optionGroupBox);
    mainLayout->addLayout(buttonsVLayout);
    setLayout(mainLayout);

    // signal slot using defaults

    // slots and signals: ATTENTION: createButton() sets signal-slot for pushButtons
    connect(fromFileRadioButton,SIGNAL(toggled(bool)),this,SLOT(readImgFromFiles()));
    connect(fromCamRadioButton,SIGNAL(toggled(bool)),this,SLOT(openCam()));
    connect(noSourceRadioButton,SIGNAL(toggled(bool)),this,SLOT(noSourceSelected()));

    connect(camTimer,SIGNAL(timeout()),this,SLOT(camTimerUp()));



    // other
    cvImageVector.clear();
    cvCcsVector.clear();
    patternFoundVector.clear();

    cvBoardSize.width = 4;
    cvBoardSize.height = 11;

    cameraMatrix.create(3,3,CV_64FC1); // double, opencv documentation: wrong, check code
    distCoeffs.create(5,1,CV_64FC1); // double, check opencv source code

    needSave = false;


    flipImageBool = false;



camAreOpen = false;





}



QPushButton *caliCam::createButton(const QString &text, QWidget *receiver,
                                   const char *member)
{
    QPushButton *button = new QPushButton(text);
    button->connect(button, SIGNAL(clicked()), receiver, member);
    return button;
}



void caliCam::saveImages(){

    QString message
            = QString("first, select a dir to save files, ")+
            "then, give a name and files will be name.0.bmp, "+
            "name.1.bmp, etc. Good to manually delete bmp files "+
            "in the dir before saving";

    QMessageBox::information(this,"info", message);

    QFileDialog::Options options
            = QFileDialog::DontResolveSymlinks | QFileDialog::ShowDirsOnly;

    QString directory
            = QFileDialog::getExistingDirectory(
                this, tr("select dir to save images"), "../",   options);

    if (directory.isEmpty()) return;

    QString fileName =
            QInputDialog::getText(
                this,
                tr("give name for files"), // title
                tr("file name:"),   // label
                QLineEdit::Normal,
                tr("use either \"eye\" or \"scene\"")
                );

    if (fileName.isEmpty()) return;

    for (unsigned int i = 0; i< cvImageVector.size();i++){
        QString fullName = directory + '/' + fileName + "." + QString::number(i) + ".bmp";
        qDebug()<<endl<<fullName;
        std::string tmpStr = fullName.toStdString();
        cv::imwrite(tmpStr.c_str(), cvImageVector[i]);
    }


    message = QString::number(cvImageVector.size()) + "files saved to " +
            directory + ", with names : "+ fileName+".[n].bmp";
    QMessageBox::information(this,
                             "file saved",
                             message);

    needSave = false;

}



void caliCam::openCam(){  // triggered by signal of selecting the item in option box

    if (!fromCamRadioButton->isChecked()) return;

    fromFileRadioButton->setDisabled(true);
    fromCamRadioButton->setDisabled(true);
    noSourceRadioButton->setDisabled(true);

    closeCamPushButton->setDisabled(true);
    grabCamImagePushButton->setDisabled(true);
    saveImagesPushButton->setDisabled(true);

    QMessageBox::
            information(
                this,"reminder",
                "going to open camera, make sure cam parameters (focus, etc) are correct");

    int cvCamId = QInputDialog::getInt(this, tr("cam id"),
                                       tr("/dev/video"), 1, 0, 4, 1);


    cvVc.open(cvCamId);
    if (!cvVc.isOpened()){
        QMessageBox::information(this,"error","cam not openned");
        fromFileRadioButton->setEnabled(true);
        fromCamRadioButton->setEnabled(true);
        noSourceRadioButton->setEnabled(true);
        noSourceRadioButton->setChecked(true);
        return;
    }

    closeCamPushButton->setEnabled(true);
    grabCamImagePushButton->setEnabled(true);

    // info: image size
    cv::Mat tmpMat;
    cvVc>>tmpMat;
    QString tmp =
            "cam image size: " +
            QString::number(tmpMat.cols) +
            "x"+
            QString::number(tmpMat.rows) +
            ", display area size: " +
            QString::number(displayLabel->size().width()) +
            "x" +
            QString::number(displayLabel->size().height());

    imgSizeLabel->setText(tmp);




    // flip or not
    int ret = QMessageBox::question(
                this,
                "flip source image",
                "flip camera image?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes) flipImageBool = true;
    else flipImageBool = false;





    // auto grab or not
    useAutoGrab = false; // default
    ret = QMessageBox::question(
                this,
                "auto grab cam image",
                "enable cam image auto grab?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes){
        useAutoGrab = true;
        ag = new AutoGrab;

        ag->ini();
    }

    // start timer
    camTimer->start(30);



    Q_ASSERT(!camAreOpen);
    camAreOpen = true;

}

void caliCam::closeCam(){

    if (!camAreOpen) return;

    cvVc.release();

    camTimer->stop();

    closeCamPushButton->setDisabled(true);
    grabCamImagePushButton->setDisabled(true);
    saveImagesPushButton->setEnabled(true);

    fromFileRadioButton->setEnabled(true);
    fromCamRadioButton->setEnabled(true);
    noSourceRadioButton->setEnabled(true);
    noSourceRadioButton->setChecked(true);


    displayLabel->setText("cam closed, no image show here");
    imgSizeLabel->clear();

    delete ag;
}


void caliCam::camTimerUp(){
    cv::Mat cvCamImage;
    // this cv::mat image is isolated from that used to save cam image into std::vector,
    // this one is used only here in this function to draw image on GUI

    std::vector<cv::Point2f> cvCc; // cc: circle centers // check source code, need floating point

    cvVc>>cvCamImage;

    // flip
    if (flipImageBool) {
        cv::flip(cvCamImage, cvCamImage, 0);
        cv::flip(cvCamImage, cvCamImage, 1);
    }

    patternFound = cv::findCirclesGrid(
                cvCamImage,
                cvBoardSize,
                cvCc,
                cv::CALIB_CB_ASYMMETRIC_GRID);



    bool captureNow = false;
    if(useAutoGrab)
    {



        cv::Scalar pa = cv::Scalar(0.0f, 0.0f); // pa: points average

        if (patternFound) pa = cv::mean(cvCc); // pa means point average

        captureNow  = ag->pushAndCheck(//myTime->elapsed(),
                                       patternFound,pa[0],pa[1]);

        if (captureNow)        cvCamImage = cv::Mat::zeros(cvCamImage.size(), cvCamImage.type());

        ag->writeInfoOnImage(cvCamImage);


    }


    if (patternFound) cv::drawChessboardCorners(cvCamImage,cvBoardSize,cv::Mat(cvCc),patternFound);
    QImage image(cvCamImage.ptr(),cvCamImage.cols,cvCamImage.rows,QImage::Format_RGB888);
    image = image.rgbSwapped();
    displayLabel->setPixmap(QPixmap::fromImage(image));

    if (captureNow) grabCamImg2ListWidget();

}

void caliCam::noSourceSelected(){
    if (noSourceRadioButton->isChecked()){
        displayLabel->setText("no image source selected");
    }
}

void caliCam::grabCamImg2ListWidget(){
    std::vector<cv::Point2f> cvCc; // cc: circle centers
    cv::Mat cvCamImage;

    cvVc>>cvCamImage;
    if (flipImageBool) {
        cv::flip(cvCamImage, cvCamImage, 0);
        cv::flip(cvCamImage, cvCamImage, 1);
    }


    cvImageVector.push_back(cvCamImage.clone());


    patternFound = cv::findCirclesGrid(cvCamImage,cvBoardSize,cvCc,cv::CALIB_CB_ASYMMETRIC_GRID);

    patternFoundVector.push_back(patternFound);
    cvCcsVector.push_back(cvCc);

    if (patternFound) cv::drawChessboardCorners(cvCamImage,cvBoardSize,cv::Mat(cvCc),patternFound);

    static int index = 0;

    QImage image(cvCamImage.ptr(),cvCamImage.cols,cvCamImage.rows,QImage::Format_RGB888);
    image = image.rgbSwapped();

    QListWidgetItem * itm = new QListWidgetItem;
    itm->setText("frame " + QString::number(index++));
    itm->setIcon(QIcon(QPixmap::fromImage(image)));

    //imageListWidget->insertItem(0,itm);
    imageListWidget->addItem(itm);
    imageListWidget->setCurrentItem(itm);

    noiLabel->setText("number of images in aobve list: "
                      + QString::number(imageListWidget->count()));

    //saveImagesPushButton->setEnabled(true);
    removeImgPushButton->setEnabled(true);
    removeBadImagesPushButton->setEnabled(true);
    needSave = true;
    calibPushButton->setEnabled(true);
}


void caliCam::removeImgFromListWidget(){

    int current_row = imageListWidget->currentRow();
    int current_count = imageListWidget->count();

    if (current_count>0) {

        if (current_row ==-1)
            QMessageBox::information(this,"error",".count > 0 but .currentRow = -1, 1342");

        delete imageListWidget->takeItem(current_row);
        cvImageVector.erase(cvImageVector.begin()+current_row);
        cvCcsVector.erase(cvCcsVector.begin()+current_row);
        patternFoundVector.erase(patternFoundVector.begin()+current_row);
        // qlistwidget save images from let to right, leftmost index is zero, and std::vector
        // append latest image to the end, the largest index number

        noiLabel->setText("number of images in aobve list: "+
                          QString::number(current_count-1));

        if (current_count == 1) // will be empty after deletion
        {
            saveImagesPushButton->setDisabled(true);
            removeImgPushButton->setDisabled(true);
        }
    }


}



void caliCam::readImgFromFiles(){

    // triggered by selecting the item from option box

    if (!fromFileRadioButton->isChecked()) return;

    // flip or not
    int ret = QMessageBox::question(
                this,
                "flip source image",
                "flip image read from file?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes) flipImageBool = true;
    else flipImageBool = false;

    std::vector<cv::Point2f> cvCc; // cc: circle centers



    if (needSave){
        int ret = QMessageBox::warning(
                    this,
                    "images not saved",
                    "the image list will be cleared, do you wanto save unsaved images?",
                    QMessageBox::No|QMessageBox::Save);
        if (ret == QMessageBox::Save) saveImages();
    }

    cvImageVector.clear();
    cvCcsVector.clear();
    patternFoundVector.clear();
    imageListWidget->clear();

    closeCam();

    QFileDialog::Options options;
    QString selectedFilter;
    QString openFilesPath;

    QStringList files = QFileDialog::getOpenFileNames(
                this, tr("select BMP image files"),
                openFilesPath,
                tr("BMP image Files (*.bmp)"),
                &selectedFilter,
                options);
    std::string tmpStr;
    QString qTmpStr;

    if (files.count()) {
        // reminder, keep std::vector cv::Mat and the list synchronized
        for ( int i = 0 ; i<files.size(); i++){
            qTmpStr = files[i];
            tmpStr = qTmpStr.toStdString();
            cv::Mat cvFileImage = cv::imread(tmpStr, 1);

            // flip
            if (flipImageBool) {
                cv::flip(cvFileImage, cvFileImage, 0);
                cv::flip(cvFileImage, cvFileImage, 1);
            }

            cvImageVector.push_back(cvFileImage.clone());


            patternFound = cv::findCirclesGrid(cvFileImage,cvBoardSize,cvCc,cv::CALIB_CB_ASYMMETRIC_GRID);

            patternFoundVector.push_back(patternFound);
            cvCcsVector.push_back(cvCc);
            if (patternFound) cv::drawChessboardCorners(cvFileImage,cvBoardSize,cv::Mat(cvCc),patternFound);


            QImage image(cvFileImage.ptr(),cvFileImage.cols,cvFileImage.rows,QImage::Format_RGB888);
            image = image.rgbSwapped();

            QListWidgetItem * itm = new QListWidgetItem;
            itm->setText("frame " + QString::number(i));
            itm->setIcon(QIcon(QPixmap::fromImage(image)));

            imageListWidget->addItem(itm);
            imageListWidget->setCurrentItem(itm);

            noiLabel->setText("number of images in aobve list: "
                              + QString::number(imageListWidget->count()));
        }
        saveImagesPushButton->setEnabled(true);
        removeImgPushButton->setEnabled(true);
        removeBadImagesPushButton->setEnabled(true);
        calibPushButton->setEnabled(true);
    }
}


void caliCam::calibrate(){

    closeCam();

    // check, if there are images contains bad pattern in the list
    for (unsigned int i = 0; i< patternFoundVector.size(); i++){
        if(! patternFoundVector[i]){
            QMessageBox::warning(this,"bad images", "bad images found in list, remove them first");
            return;
        }
    }

    int numOfImages = imageListWidget->count();

//    if (numOfImages<1 || numOfImages>50) {
//        int ret = QMessageBox::information(this,"number of images", "too less or too many images for calibration", QMessageBox::Cancel|QMessageBox::Ok);
//        if (ret == QMessageBox::Cancel) return;
//    }

    std::vector<cv::Point3f>  cvObjectPoints; // points for the chess board for one view // floating point required, check opencv source code

    // make points for one view, and a vector of multiple views
    for (int i=0; i<cvBoardSize.height; i++) // outer iterate, row by row
        for (int j=0; j<cvBoardSize.width; j++) // inner iteration, colum by colum
            cvObjectPoints.push_back(cv:: Point3f( // floating point required, check opencv source code
                                                   float((2*j + i % 2)*1),
                                                   float(i*1),
                                                   0
                                                   )
                                     );
    std::vector<std::vector< cv::Point3f > > cvObjectPointsVector; // multiple views // floating point required, check opencv source code
    for (int i=0; i<numOfImages; i++) cvObjectPointsVector.push_back(cvObjectPoints); // for all the views, save 3D object points

    displayLabel->clear();
    displayLabel->setText("calibrating...");

    cv::Size cvImageSize;
    cvImageSize = cvImageVector[0].size();




    double rms = cv::calibrateCamera (cvObjectPointsVector,cvCcsVector,cvImageSize,cameraMatrix, distCoeffs,rvecs,tvecs);





    Q_ASSERT(cameraMatrix.size()==cv::Size(3,3));
    Q_ASSERT(cameraMatrix.type()==CV_64FC1);

    Q_ASSERT(rvecs[0].size()==cv::Size(1,3)); // mat::Size() returns colsxrows!
    Q_ASSERT(rvecs[0].type()==CV_64FC1);
    Q_ASSERT(tvecs[0].size()==cv::Size(1,3));// mat::Size() returns colsxrows!
    Q_ASSERT(tvecs[0].type()==CV_64FC1);



    displayLabel->clear();
    displayLabel->setText(QString("calib says rms is: %1").arg(rms));

    saveParametersPushButton->setEnabled(true);

}

void caliCam::cleanImageList(){


    // removing
    for (int i= patternFoundVector.size()-1; i>=0; i--){ // important, bigger to smaller index

        if(! patternFoundVector[i]){

            delete imageListWidget->takeItem(i);
            cvImageVector.erase(cvImageVector.begin()+i);
            cvCcsVector.erase(cvCcsVector.begin()+i);
            patternFoundVector.erase(patternFoundVector.begin()+i);
            // qlistwidget save images from let to right, leftmost index is zero, and std::vector
            // append latest image to the end, the largest index number

            noiLabel->setText("number of images in aobve list: "+ QString::number(imageListWidget->count()));

            if (!imageListWidget->count()) // will be empty after deletion
            {
                saveImagesPushButton->setDisabled(true);
                removeImgPushButton->setDisabled(true);
            }

        }
    }

}





void caliCam::saveParameters(){
    QFileDialog::Options options =  QFileDialog::ShowDirsOnly;
    QString directory = QFileDialog::getExistingDirectory(this,
                                                          tr("select dir to save cali result"),
                                                          "../",
                                                          options);
    if (directory.isEmpty()) return;

    QString QStringTmp;

    std::stringstream ss;
    QStringTmp = directory + "/cam.txt";
    std::string fileName = QStringTmp.toStdString();
    std::ofstream camFile (fileName.c_str());

    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);


    if (camFile.is_open())
    {
        for (int i=0; i<3; i++){
            ss.clear();
            ss.str("");
            for (int j=0; j<3; j++){
                ss<<cameraMatrix.at<double>(i,j)<<'\t';
                // double, opencv doc wrong, check code
            }
            camFile<<ss.str()<<'\n';
        }
        camFile.close();
    }
    else qDebug()<< "Unable to open file cam.txt"<<endl;

    // save it again using opencv

    QStringTmp = directory +"/cam.yaml";
    fileName = QStringTmp.toStdString();
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    fs<<"intrinsic"<<cameraMatrix;
    fs.release();




    // save distortion coeff
    QStringTmp = directory + "/dst.txt";
    fileName = QStringTmp.toStdString();
    std::ofstream dstFile (fileName.c_str());

    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);

    if (dstFile.is_open())
    {
        ss.clear();
        ss.str("");
        for (int i=0; i<5; i++) ss<<distCoeffs.at<double>(i,0)<<'\t';
        dstFile<<ss.str();
        dstFile.close();
    }
    else qDebug() << "Unable to open file dst.txt"<<endl;

    // save it again using opencv
    QStringTmp = directory + "/dst.yaml";
    fileName = QStringTmp.toStdString();

    cv::FileStorage fs2(fileName, cv::FileStorage::WRITE);
    fs2<<"distortion"<<distCoeffs;
    fs2.release();


    // save rvecs
    QStringTmp = directory + "/rvecs.txt";
    fileName = QStringTmp.toStdString();
    std::ofstream rvecsFile (fileName.c_str());

    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);

    if (rvecsFile.is_open())
    {
        for (int unsigned i= 0; i<rvecs.size(); i++) {


            ss.clear();
            ss.str("");
            ss
                    <<rvecs[i].at<double>(0,0)<<'\t'
                   <<rvecs[i].at<double>(1,0)<<'\t'
                  <<rvecs[i].at<double>(2,0);
            rvecsFile<<ss.str();
            rvecsFile<<'\n';
        }
        rvecsFile.close();
    }
    else qDebug() << "Unable to open file rvecs.txt"<<endl;




    // save tvecs
    QStringTmp = directory + "/tvecs.txt";
    fileName = QStringTmp.toStdString();
    std::ofstream tvecsFile (fileName.c_str());

    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);

    if (tvecsFile.is_open())
    {
        for (unsigned int i= 0; i<tvecs.size(); i++) {
            ss.clear();
            ss.str("");
            ss
                    <<tvecs[i].at<double>(0,0)<<'\t'
                   <<tvecs[i].at<double>(1,0)<<'\t'
                  <<tvecs[i].at<double>(2,0);
            tvecsFile<<ss.str();
            tvecsFile<<'\n';
        }
        tvecsFile.close();
    }
    else qDebug() << "Unable to open file tvecs.txt"<<endl;



    // save rmats
    cv::Mat rmat;
    QStringTmp = directory + "/rmats.txt";
    fileName = QStringTmp.toStdString();
    std::ofstream rmatsFile (fileName.c_str());

    ss.flags(std::ios::scientific);
    ss.precision(std::numeric_limits<double>::digits10);

    if (rmatsFile.is_open())
    {
        for ( unsigned int k= 0; k<rvecs.size(); k++) {
            cv::Rodrigues(rvecs[k], rmat);
            Q_ASSERT(rmat.size()==cv::Size(3,3));
            Q_ASSERT(rmat.type()==CV_64FC1);
            for (int i=0; i<3; i++)
            {
                ss.clear();
                ss.str("");
                ss
                        <<rmat.at<double>(i,0)<<'\t'
                       <<rmat.at<double>(i,1)<<'\t'
                      <<rmat.at<double>(i,2);
                rmatsFile<<ss.str();
                rmatsFile<<'\n';
            }
        }
        rmatsFile.close();
    }
    else qDebug() << "Unable to open file rmats.txt"<<endl;



}


void caliCam::setCamDisplaySize(int w, int h){
    displayLabel->setMinimumSize(w,h);
    displayLabel->setMaximumSize(w,h);
}


void caliCam::hideCalibButtons(){

    buttonsHLayout3->removeWidget(calibPushButton);
    buttonsHLayout3->removeWidget(saveParametersPushButton);
    delete calibPushButton;
    delete saveParametersPushButton;
    buttonsVLayout->removeItem(buttonsHLayout3);

}
