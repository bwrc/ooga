#include "twocam.h"

#include <QtGui>

#include <QGridLayout>
#include <QListWidget>




void TwoCam::cleanImageList(){

    // removing
    for (int i= eyePatternFoundVector.size()-1; i>=0; i--)
    { // important, bigger to smaller index

        if(
                ! eyePatternFoundVector[i] ||
                ! scenePatternFoundVector[i]
                ){

            delete eyeImageListWidget->takeItem(i);
            delete sceneImageListWidget->takeItem(i);

            cvEyeImageVector.erase(cvEyeImageVector.begin()+i);
            cvSceneImageVector.erase(cvSceneImageVector.begin()+i);

            cvEyeCcsVector.erase(cvEyeCcsVector.begin()+i);
            cvSceneCcsVector.erase(cvSceneCcsVector.begin()+i);

            eyePatternFoundVector.erase(eyePatternFoundVector.begin()+i);
            scenePatternFoundVector.erase(scenePatternFoundVector.begin()+i);
            // qlistwidget save images from let to right, leftmost index is zero, and std::vector
            // append latest image to the end, the largest index number

            noiLabel->setText
                    ("number of images in aobve list: "+
                     QString::number(eyeImageListWidget->count()));

            if (!eyeImageListWidget->count()) // will be empty after deletion
            {
                saveImagesPushButton->setDisabled(true);
                removeEyeImgPushButton->setDisabled(true);
                removeSceneImgPushButton->setDisabled(true);
            }

        }
    }

}



QPushButton *TwoCam::createButton(const QString &text, QWidget *receiver,
                                  const char *member)
{
    QPushButton *button = new QPushButton(text);
    button->connect(button, SIGNAL(clicked()), receiver, member);
    return button;
}





TwoCam::TwoCam(QWidget *parent) :
    QWidget(parent)
{

    eyeTopLabel = new QLabel;
    eyeTopLabel->setText("virutal EYE cam");
    eyeTopLabel->setAlignment(Qt::AlignHCenter);

    sceneTopLabel = new QLabel;
    sceneTopLabel->setText("SCENE cam");
    sceneTopLabel->setAlignment(Qt::AlignHCenter);

    eyeDisplayLabel = new QLabel;
    eyeDisplayLabel->setMaximumSize(320,240);
    eyeDisplayLabel->setMinimumSize(320,240);
    eyeDisplayLabel->setText("eye cam not open");
    eyeDisplayLabel->setAlignment(Qt::AlignHCenter);
    eyeDisplayLabel->setScaledContents(true);


    sceneDisplayLabel = new QLabel;
    sceneDisplayLabel->setMaximumSize(320,240);
    sceneDisplayLabel->setMinimumSize(320,240);
    sceneDisplayLabel->setText("scene cam not open");
    sceneDisplayLabel->setAlignment(Qt::AlignHCenter);
    sceneDisplayLabel->setScaledContents(true);

    eyeImageListWidget = new QListWidget;
    eyeImageListWidget->setViewMode(QListView::IconMode);
    eyeImageListWidget->setIconSize(QSize(128,96));
    eyeImageListWidget->setMovement(QListView::Free);
    eyeImageListWidget->setMinimumWidth(128*3);
    eyeImageListWidget->setSpacing(12);
    eyeImageListWidget->setFlow(QListWidget::LeftToRight);


    sceneImageListWidget = new QListWidget;
    sceneImageListWidget->setViewMode(QListView::IconMode);
    sceneImageListWidget->setIconSize(QSize(128,96));
    sceneImageListWidget->setMovement(QListView::Free);
    sceneImageListWidget->setMinimumWidth(128*3);
    sceneImageListWidget->setSpacing(12);
    sceneImageListWidget->setFlow(QListWidget::LeftToRight);

    noiLabel = new QLabel("number of images in above list: 0");

    optionGroupBox = new QGroupBox("select image source");
    fromFileRadioButton = new QRadioButton;
    fromFileRadioButton->setText("load image files");
    fromCamRadioButton = new QRadioButton;
    fromCamRadioButton->setText("grab images from cam");
    noSourceRadioButton = new QRadioButton;
    noSourceRadioButton->setText("nothing to select");
    noSourceRadioButton->setChecked(true);

    // creat buttons

    saveImagesPushButton = createButton("save images in list",this,SLOT(saveImages()));
    saveImagesPushButton->setDisabled(true);
    closeCamPushButton = createButton("close cam",this,SLOT(closeCam()));
    closeCamPushButton->setDisabled(true);
    grabCamImagePushButton = createButton("grab cam image",this, SLOT(grabCamImage2ListWidget()));
    grabCamImagePushButton->setDisabled(true);
    removeEyeImgPushButton = createButton("remove current EYE img from list",this,SLOT(removeEyeImageFromListWidget()));
    removeEyeImgPushButton->setDisabled(true);
    removeSceneImgPushButton = createButton("remove current SCENE img from list",this,SLOT(removeSceneImageFromListWidget()));
    removeSceneImgPushButton->setDisabled(true);
    removeBadImagesPushButton = createButton("remove bad images",this,SLOT(cleanImageList()));
    removeBadImagesPushButton->setDisabled(true);


    QGridLayout * mainLayout = new QGridLayout;
    mainLayout->addWidget(eyeTopLabel,0,0);
    mainLayout->addWidget(sceneTopLabel,0,1);
    mainLayout->addWidget(eyeDisplayLabel,1,0);
    mainLayout->addWidget(sceneDisplayLabel,1,1);
    mainLayout->addWidget(eyeImageListWidget,2,0);
    mainLayout->addWidget(sceneImageListWidget,2,1);
    mainLayout->addWidget(removeEyeImgPushButton,3,0);
    mainLayout->addWidget(removeSceneImgPushButton,3,1);
    mainLayout->addWidget(noiLabel, 4,0,1,2);

    optionLayout = new QHBoxLayout;
    optionLayout->addWidget(fromFileRadioButton);
    optionLayout->addWidget(fromCamRadioButton);
    optionLayout->addWidget(noSourceRadioButton);
    optionGroupBox->setLayout(optionLayout);


    mainLayout->addWidget(optionGroupBox, 5,0,1,2);


    // buttons layout


    buttonsHLayout = new QHBoxLayout;

    buttonsHLayout->addWidget(grabCamImagePushButton);
    buttonsHLayout->addWidget(closeCamPushButton);


    buttonsHLayout->addWidget(removeBadImagesPushButton);
    buttonsHLayout->addWidget(saveImagesPushButton);

    mainLayout->addLayout(buttonsHLayout,6,0,1,2);





    setLayout(mainLayout);


    // timer
    camTimer = new QTimer;

    // connections
    // slots and signals: ATTENTION: createButton() sets signal-slot for pushButtons
    connect(fromFileRadioButton,SIGNAL(toggled(bool)),this,SLOT(readImageFiles()));
    connect(fromCamRadioButton,SIGNAL(toggled(bool)),this,SLOT(openCam()));
    connect(noSourceRadioButton,SIGNAL(toggled(bool)),this,SLOT(noSourceSelected()));

    connect(camTimer,SIGNAL(timeout()),this,SLOT(camTimerUp()));

    // other
    cvEyeImageVector.clear();
    cvSceneImageVector.clear();

    cvEyeCcsVector.clear();
    cvSceneCcsVector.clear();

    eyePatternFoundVector.clear();
    scenePatternFoundVector.clear();

    cvBoardSize.width = 4;
    cvBoardSize.height = 11;

    needSave = false;
    flipImageBool = false;


    useAutoGrab = false;

    camAreOpen = false;

}


TwoCam::~TwoCam(){
    if (!cvEyeVc.isOpened())  cvEyeVc.release();
    if (!cvSceneVc.isOpened())  cvSceneVc.release();
    camTimer->stop();
}


void TwoCam::openCam(){// triggered by signal of selecting the item in option box

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
                "going to open EYE camera, make sure cam parameters (focus, etc) are correct");

    int camId = QInputDialog::getInt(this, tr("cam id for EYE cam"),
                                     tr("EYE cam: /dev/video"), 1, 0, 4, 1);


    // flip or not
//    int ret = QMessageBox::question(
//                this,
//                "flip source image for eye cam",
//                "flip EYE camera image?",
//                QMessageBox::Yes | QMessageBox::No);
//    if (ret == QMessageBox::Yes) flipImageBool = true;
//    else flipImageBool = false;


    cvEyeVc.open(camId);
    if (!cvEyeVc.isOpened()){
        QMessageBox::information(this,"error","EYE cam not openned");
        fromFileRadioButton->setEnabled(true);
        fromCamRadioButton->setEnabled(true);
        noSourceRadioButton->setEnabled(true);
        noSourceRadioButton->setChecked(true);
        return;
    }


    QMessageBox::
            information(
                this,"reminder",
                "going to open SCENE camera, make sure cam parameters (focus, etc) are correct");

    camId = QInputDialog::getInt(this, tr("cam id for SCENE cam"),
                                 tr("SCENE cam: /dev/video"), 2, 0, 4, 1);

    // flip or not
    int ret = QMessageBox::question(
                this,
                "flip the images for scene camera",
                "Flip scene camera images?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes) flipImageBool = true;
    else flipImageBool = false;

    cvSceneVc.open(camId);
    if (!cvSceneVc.isOpened()){
        QMessageBox::information(this,"error","SCENE cam not openned");
        fromFileRadioButton->setEnabled(true);
        fromCamRadioButton->setEnabled(true);
        noSourceRadioButton->setEnabled(true);
        noSourceRadioButton->setChecked(true);
        return;
    }


    closeCamPushButton->setEnabled(true);
    grabCamImagePushButton->setEnabled(true);




    // auto grab or not
    useAutoGrab = false; // default
    ret = QMessageBox::question(
                this,
                "auto grab cam image",
                "enable cam image auto grab?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes){
        useAutoGrab = true;
        ag = new AutoGrabTwoCam;
        ag->ini();
    }



    camTimer->start(//33);
                    20
                    );



    // info : image size
    cv::Mat tmpMat;
    cvEyeVc>>tmpMat;
    QString tmp =
            "Eye cam image size: " +
            QString::number(tmpMat.cols) +
            "x"+
            QString::number(tmpMat.rows) +
            ", display area size: " +
            QString::number(eyeDisplayLabel->size().width()) +
            "x" +
            QString::number(eyeDisplayLabel->size().height());

    eyeTopLabel->setText(tmp);

    cvSceneVc>>tmpMat;
    tmp =
            "Scene cam image size: " +
            QString::number(tmpMat.cols) +
            "x"+
            QString::number(tmpMat.rows) +
            ", display area size: " +
            QString::number(sceneDisplayLabel->size().width()) +
            "x" +
            QString::number(sceneDisplayLabel->size().height());

    sceneTopLabel->setText(tmp);




    Q_ASSERT(!camAreOpen);
    camAreOpen = true;

}



void TwoCam::closeCam(){

    if (!camAreOpen) return;

    cvEyeVc.release();
    cvSceneVc.release();

    camTimer->stop();

    closeCamPushButton->setDisabled(true);
    grabCamImagePushButton->setDisabled(true);
    saveImagesPushButton->setEnabled(true);

    fromFileRadioButton->setEnabled(true);
    fromCamRadioButton->setEnabled(true);
    noSourceRadioButton->setEnabled(true);
    noSourceRadioButton->setChecked(true);


    eyeTopLabel->clear();
    eyeTopLabel->setText("eye cam closed");
    sceneTopLabel->clear();
    sceneTopLabel->setText("scene cam closed");


    delete ag;
    // cause problem if you call closeCam without corresponding openCam
    // so, code added to make sure cam are opened, closed, opened, closed, ...
}


void TwoCam::camTimerUp(){

    cv::Mat cvEyeCamImage;
    cv::Mat cvSceneCamImage;

    cvEyeVc>>cvEyeCamImage;
    cvSceneVc>>cvSceneCamImage;

    if (flipImageBool) {
        cv::flip(cvSceneCamImage, cvSceneCamImage, 0);
        cv::flip(cvSceneCamImage, cvSceneCamImage, 1);
    }

    // for eye cam

    eyePatternFound = cv::findCirclesGrid(cvEyeCamImage,
                                          cvBoardSize,
                                          cvEyeCcs,
                                          cv::CALIB_CB_ASYMMETRIC_GRID);

    //std::cout << "cvEyeCcs:  " << std::endl << cvEyeCcs << std::endl;



    // for scene cam

    scenePatternFound = cv::findCirclesGrid(cvSceneCamImage,
                                            cvBoardSize,
                                            cvSceneCcs,
                                            cv::CALIB_CB_ASYMMETRIC_GRID);




    bool captureNow = false;
    if(useAutoGrab)
    {
        cv::Scalar eyePa, scenePa; // pa: points average
        eyePa = cv::Scalar(0.0f, 0.0f);
        scenePa = cv::Scalar(0.0f, 0.0f);
        if (eyePatternFound) eyePa = cv::mean(cvEyeCcs); // pa means point average
        if (scenePatternFound) scenePa = cv::mean(cvSceneCcs); // pa means point average


        captureNow = ag->pushAndCheck(
                    eyePatternFound,
                    scenePatternFound,
                    eyePa[0],
                    scenePa[0],
                    eyePa[1],
                    scenePa[1]
                    );

        if (captureNow) {
            cvEyeCamImage = cv::Mat::zeros(cvEyeCamImage.size(),cvEyeCamImage.type());
            cvSceneCamImage = cv::Mat::zeros(cvSceneCamImage.size(),cvSceneCamImage.type());
        }

        ag->writeInfoOnImage(cvEyeCamImage,cvSceneCamImage);

    }

    if (eyePatternFound)
        cv::drawChessboardCorners(
                    cvEyeCamImage,
                    cvBoardSize,
                    cv::Mat(cvEyeCcs),
                    eyePatternFound);
    if (scenePatternFound)
        cv::drawChessboardCorners(
                    cvSceneCamImage,
                    cvBoardSize,
                    cv::Mat(cvSceneCcs),
                    scenePatternFound);

    QImage eyeImage(cvEyeCamImage.ptr(),
                    cvEyeCamImage.cols,
                    cvEyeCamImage.rows,
                    QImage::Format_RGB888);
    eyeImage = eyeImage.rgbSwapped();
    eyeDisplayLabel->setPixmap(QPixmap::fromImage(eyeImage));

    QImage sceneImage(cvSceneCamImage.ptr(),
                      cvSceneCamImage.cols,
                      cvSceneCamImage.rows,
                      QImage::Format_RGB888);
    sceneImage = sceneImage.rgbSwapped();
    sceneDisplayLabel->setPixmap(QPixmap::fromImage(sceneImage));

    if (captureNow)           grabCamImage2ListWidget();
}




void TwoCam::noSourceSelected(){
    if (noSourceRadioButton->isChecked()){
        eyeDisplayLabel->setText("no image source selected for Eye cam");
        sceneDisplayLabel->setText("no image source selected for SCENE cam");
    }


}

void TwoCam::grabCamImage2ListWidget(){

    //qDebug()<<"entering grabCamImage2List";

    static int index = 0;



    cv::Mat cvEyeCamImage;
    cv::Mat cvSceneCamImage;

    cvEyeVc>>cvEyeCamImage;
    cvSceneVc>>cvSceneCamImage;

    if (flipImageBool) {
        cv::flip(cvSceneCamImage, cvSceneCamImage, 0);
        cv::flip(cvSceneCamImage, cvSceneCamImage, 1);
    }




    // to eye list
    cvEyeImageVector.push_back(cvEyeCamImage.clone());

    eyePatternFound = cv::findCirclesGrid(
                cvEyeCamImage,
                cvBoardSize,
                cvEyeCcs,
                cv::CALIB_CB_ASYMMETRIC_GRID);
    eyePatternFoundVector.push_back(eyePatternFound);
    cvEyeCcsVector.push_back(cvEyeCcs);

    if (eyePatternFound) cv::drawChessboardCorners(
                cvEyeCamImage,
                cvBoardSize,
                cv::Mat(cvEyeCcs),
                eyePatternFound);


    QImage eyeImage(
                cvEyeCamImage.ptr(),
                cvEyeCamImage.cols,
                cvEyeCamImage.rows,
                QImage::Format_RGB888);
    eyeImage = eyeImage.rgbSwapped();

    QListWidgetItem * eyeItm = new QListWidgetItem;
    eyeItm->setText("frame " + QString::number(index));
    eyeItm->setIcon(QIcon(QPixmap::fromImage(eyeImage)));


    eyeImageListWidget->addItem(eyeItm);
    eyeImageListWidget->setCurrentItem(eyeItm);


    // to scene list
    cvSceneImageVector.push_back(cvSceneCamImage.clone());

    scenePatternFound = cv::findCirclesGrid(
                cvSceneCamImage,
                cvBoardSize,
                cvSceneCcs,
                cv::CALIB_CB_ASYMMETRIC_GRID);
    scenePatternFoundVector.push_back(scenePatternFound);
    cvSceneCcsVector.push_back(cvSceneCcs);

    if (scenePatternFound) cv::drawChessboardCorners(
                cvSceneCamImage,
                cvBoardSize,
                cv::Mat(cvSceneCcs),
                scenePatternFound);


    QImage sceneImage(
                cvSceneCamImage.ptr(),
                cvSceneCamImage.cols,
                cvSceneCamImage.rows,
                QImage::Format_RGB888);
    sceneImage = sceneImage.rgbSwapped();

    QListWidgetItem * sceneItm = new QListWidgetItem;
    sceneItm->setText("frame " + QString::number(index));
    sceneItm->setIcon(QIcon(QPixmap::fromImage(sceneImage)));


    sceneImageListWidget->addItem(sceneItm);
    sceneImageListWidget->setCurrentItem(sceneItm);





    // common part for two cams

    removeEyeImgPushButton->setEnabled(true);
    removeSceneImgPushButton->setEnabled(true);
    removeBadImagesPushButton->setEnabled(true);
    needSave = true;

    noiLabel->setText("number of images in aobve list: "
                      + QString::number(eyeImageListWidget->count()));
    index++;


    //    qDebug()<<"leaving grabCamImage2List";
}





void TwoCam::removeEyeImageFromListWidget(){
    int current_row = eyeImageListWidget->currentRow();
    int current_count = eyeImageListWidget->count();



    if (current_count>0) {

        if (current_row ==-1)
            QMessageBox::information(
                        this,
                        "error",
                        "listWidget.count() > 0 but .currentRow = -1, 1342");

        // remove item in eye list
        delete eyeImageListWidget->takeItem(current_row);
        cvEyeImageVector.erase(cvEyeImageVector.begin()+current_row);
        cvEyeCcsVector.erase(cvEyeCcsVector.begin()+current_row);
        eyePatternFoundVector.erase(eyePatternFoundVector.begin()+current_row);
        // qlistwidget save images from let to right, leftmost index is zero, and std::vector
        // append latest image to the end, the largest index number


        if (current_count == 1) // will be empty after deletion
        {
            saveImagesPushButton->setDisabled(true);
            removeEyeImgPushButton->setDisabled(true);
        }


        // remove item in scene list


        delete sceneImageListWidget->takeItem(current_row);
        cvSceneImageVector.erase(cvSceneImageVector.begin()+current_row);
        cvSceneCcsVector.erase(cvSceneCcsVector.begin()+current_row);
        scenePatternFoundVector.erase(scenePatternFoundVector.begin()+current_row);
        // qlistwidget save images from let to right, leftmost index is zero, and std::vector
        // append latest image to the end, the largest index number


        if (current_count == 1) // will be empty after deletion
        {
            saveImagesPushButton->setDisabled(true);
            removeSceneImgPushButton->setDisabled(true);
        }





        noiLabel->setText("number of images in aobve list: "+
                          QString::number(current_count-1));


    }

}




void TwoCam::removeSceneImageFromListWidget(){
    int current_row = sceneImageListWidget->currentRow();
    int current_count = sceneImageListWidget->count();

    if (current_count>0) {

        if (current_row ==-1)
            QMessageBox::information(
                        this,
                        "error",
                        "listWidget.count() > 0 but .currentRow = -1, 1342");

        // remove item in scene list
        delete sceneImageListWidget->takeItem(current_row);
        cvSceneImageVector.erase(cvSceneImageVector.begin()+current_row);
        cvSceneCcsVector.erase(cvSceneCcsVector.begin()+current_row);
        scenePatternFoundVector.erase(scenePatternFoundVector.begin()+current_row);
        // qlistwidget save images from let to right, leftmost index is zero, and std::vector
        // append latest image to the end, the largest index number


        if (current_count == 1) // will be empty after deletion
        {
            saveImagesPushButton->setDisabled(true);
            removeSceneImgPushButton->setDisabled(true);
        }

        // remove item in eye list
        delete eyeImageListWidget->takeItem(current_row);
        cvEyeImageVector.erase(cvEyeImageVector.begin()+current_row);
        cvEyeCcsVector.erase(cvEyeCcsVector.begin()+current_row);
        eyePatternFoundVector.erase(eyePatternFoundVector.begin()+current_row);
        // qlistwidget save images from let to right, leftmost index is zero, and std::vector
        // append latest image to the end, the largest index number


        if (current_count == 1) // will be empty after deletion
        {
            saveImagesPushButton->setDisabled(true);
            removeEyeImgPushButton->setDisabled(true);
        }

        noiLabel->setText("number of images in aobve list: "+
                          QString::number(current_count-1));


    }

}



void TwoCam::saveImages(){

    QString message =
            "select a dir to save files, files will be eye.0.bmp, scene.0.bmp, eye.1.bmp, scene.1.bmp, etc. Good to manually delete bmp files in the dir before saving";


    QMessageBox::information(this,"info", message);

    QFileDialog::Options options = QFileDialog::DontResolveSymlinks | QFileDialog::ShowDirsOnly;

    QString directory = QFileDialog::getExistingDirectory(this,
                                                          tr("select dir to save images"),
                                                          "../",
                                                          options);
    if (directory.isEmpty()) return;

    /*
    QString fileName =
            QInputDialog::getText(this, tr("give name for files"), tr("file name:"),
                                  QLineEdit::Normal);

    if (fileName.isEmpty()) return;
    */

    for (unsigned int i = 0; i< cvEyeImageVector.size();i++){
        QString fullName;
        std::string tmpStr;

        // save eye image
        fullName = directory + '/' + "eye." + QString::number(i) + ".bmp";
        //qDebug()<<endl<<fullName;
        tmpStr = fullName.toStdString();
        cv::imwrite(tmpStr.c_str(), cvEyeImageVector[i]);

        // save scene image
        fullName = directory + '/' + "scene." + QString::number(i) + ".bmp";
        //qDebug()<<endl<<fullName;
        tmpStr = fullName.toStdString();
        cv::imwrite(tmpStr.c_str(), cvSceneImageVector[i]);
    }


    message = QString::number(cvEyeImageVector.size()) + "pairs of images saved to " +
            directory;
    QMessageBox::information(this,
                             "file saved",
                             message);

    needSave = false;

}



void TwoCam::readImageFiles(){



    // triggered by selecting the item from option box

    if (!fromFileRadioButton->isChecked()) return;

    closeCam();

    if (needSave){
        int ret = QMessageBox::warning(
                    this,
                    "images not saved",
                    "the image list will be cleared, do you wanto save unsaved images?",
                    QMessageBox::No|QMessageBox::Save);
        if (ret == QMessageBox::Save) saveImages();
    }

    cvEyeImageVector.clear();
    cvEyeCcsVector.clear();
    eyePatternFoundVector.clear();
    eyeImageListWidget->clear();

    cvSceneImageVector.clear();
    cvSceneCcsVector.clear();
    scenePatternFoundVector.clear();
    sceneImageListWidget->clear();



    QFileDialog::Options options;
    QString selectedFilter;
    QString openFilesPath;

    QMessageBox::information(this,
                             "open files",
                             QString("first, open eye.[n].bmp files, then, open scene.[n].bmp files")+
                             "\nATTENTION: file names pattern must be eye.[n].bmp and "
                             +"scene.[n].bmp, program will check the names");

    QStringList eyeFiles = QFileDialog::getOpenFileNames(
                this, tr("select BMP image files for eye Cam"),
                openFilesPath,
                tr("BMP image Files (eye*.bmp)"),
                &selectedFilter,
                options);

    // flip or not
    int ret = QMessageBox::question(
                this,
                "flip source image for eye cam",
                "flip EYE camera image?",
                QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes) flipImageBool = true;
    else flipImageBool = false;




    QStringList sceneFiles = QFileDialog::getOpenFileNames(
                this, tr("select BMP image files for scene Cam"),
                openFilesPath,
                tr("BMP image Files (scene*.bmp)"),
                &selectedFilter,
                options);

    // sorting and compare
    eyeFiles.sort();
    sceneFiles.sort();

    // check
    if (eyeFiles.size()!=sceneFiles.size()){
        QMessageBox::warning(this,
                             "error",
                             "# of eye image files not equal to # of scene image files");

    }


    for (int i=0;i<eyeFiles.size();i++){
        QString eyeFileName = eyeFiles.at(i);
        QString sceneFileName = sceneFiles.at(i);
        QString eyeIndex, sceneIndex;
        eyeIndex = eyeFileName.section('.',-2,-2);
        sceneIndex = sceneFileName.section('.',-2,-2);
        if (eyeIndex != sceneIndex){
            QMessageBox::warning(this,
                                 "error in reading files",
                                 QString("problem in making pairs of eye image file")+
                                 " and scene image file: "+
                                 "eye." + eyeIndex + ".bmp was read but scene."
                                 + eyeIndex + ".bmp not found"
                                 );
            return;
        }
    }

    // now, verified that, it looks like
    // eyeFiles   = eye.1.bmp   eye.3.bmp   eye.4.bmp ...
    // sceneFiles = scene.1.bmp scene.3.bmp scene.4.bmp ...
    // i.e. pairs are made successfully


    std::string tmpStr;
    QString qTmpStr;

    cv::Mat cvEyeFileImage;
    cv::Mat cvSceneFileImage;

    if (eyeFiles.count()) {
        // reminder, keep std::vector cv::Mat and the list synchronized
        for ( int i = 0 ; i<eyeFiles.size(); i++){
            qTmpStr = eyeFiles[i];
            tmpStr = qTmpStr.toStdString();
            cvEyeFileImage = cv::imread(tmpStr, 1);

            cvEyeImageVector.push_back(cvEyeFileImage.clone());


            eyePatternFound = cv::findCirclesGrid(
                        cvEyeFileImage,
                        cvBoardSize,
                        cvEyeCcs,
                        cv::CALIB_CB_ASYMMETRIC_GRID);
            eyePatternFoundVector.push_back(eyePatternFound);
            cvEyeCcsVector.push_back(cvEyeCcs);
            if (eyePatternFound)
                cv::drawChessboardCorners(
                            cvEyeFileImage,
                            cvBoardSize,
                            cv::Mat(cvEyeCcs),
                            eyePatternFound);

            QImage eyeImage(
                        cvEyeFileImage.ptr(),
                        cvEyeFileImage.cols,
                        cvEyeFileImage.rows,
                        QImage::Format_RGB888);
            eyeImage = eyeImage.rgbSwapped();

            QListWidgetItem * eyeItm = new QListWidgetItem;
            eyeItm->setText("frame " + QString::number(i));
            eyeItm->setIcon(QIcon(QPixmap::fromImage(eyeImage)));

            eyeImageListWidget->addItem(eyeItm);
            eyeImageListWidget->setCurrentItem(eyeItm);


            // repeat above steps for the other list


            qTmpStr = sceneFiles[i];
            tmpStr = qTmpStr.toStdString();
            cvSceneFileImage = cv::imread(tmpStr, 1);            
            if (flipImageBool) {
                cv::flip(cvSceneFileImage, cvSceneFileImage, 0);
                cv::flip(cvSceneFileImage, cvSceneFileImage, 1);
            }
            cvSceneImageVector.push_back(cvSceneFileImage.clone());


            scenePatternFound = cv::findCirclesGrid(
                        cvSceneFileImage,
                        cvBoardSize,
                        cvSceneCcs,
                        cv::CALIB_CB_ASYMMETRIC_GRID);
            scenePatternFoundVector.push_back(scenePatternFound);
            cvSceneCcsVector.push_back(cvSceneCcs);
            if (scenePatternFound)
                cv::drawChessboardCorners(
                            cvSceneFileImage,
                            cvBoardSize,
                            cv::Mat(cvSceneCcs),
                            scenePatternFound);

            QImage sceneImage(
                        cvSceneFileImage.ptr(),
                        cvSceneFileImage.cols,
                        cvSceneFileImage.rows,
                        QImage::Format_RGB888);
            sceneImage = sceneImage.rgbSwapped();

            QListWidgetItem * sceneItm = new QListWidgetItem;
            sceneItm->setText("frame " + QString::number(i));
            sceneItm->setIcon(QIcon(QPixmap::fromImage(sceneImage)));

            sceneImageListWidget->addItem(sceneItm);
            sceneImageListWidget->setCurrentItem(sceneItm);



        }
        noiLabel->setText("number of images in aobve list: "
                          + QString::number(eyeImageListWidget->count()));

        saveImagesPushButton->setEnabled(true);
        removeEyeImgPushButton->setEnabled(true);
        removeSceneImgPushButton->setEnabled(true);
        removeBadImagesPushButton->setEnabled(true);
    }

}

unsigned int TwoCam::getNoip(){
    return eyeImageListWidget->count();
}
