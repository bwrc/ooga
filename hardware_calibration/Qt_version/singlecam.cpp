#include<QtGui>

#include "singlecam.h"
#include <iomanip>

UseFile::UseFile(QWidget *parent): QWidget (parent)
{
    camLabel = new QLabel;
    camLabel->setText(tr("load cam.yaml for camera matrix (intrinsic parameters)"));

    dstLabel = new QLabel;
    dstLabel->setText(tr("load dst.yaml for distortion parameters"));

    loadCamFilePushButton = new QPushButton;
    loadCamFilePushButton->setText("load cam.yaml");

    loadDstFilePushButton = new QPushButton;
    loadDstFilePushButton->setText("load dst.yaml");

    camTableWidget = new QTableWidget;
    dstTableWidget = new QTableWidget;


    QHBoxLayout *camLayout = new QHBoxLayout;
    camLayout->addWidget(camLabel);
    camLayout->addWidget(loadCamFilePushButton);



    QHBoxLayout *dstLayout = new QHBoxLayout;
    dstLayout->addWidget(dstLabel);
    dstLayout->addWidget(loadDstFilePushButton);



    QVBoxLayout *mainLayout = new QVBoxLayout;

    mainLayout->addSpacing(20);
    mainLayout->addLayout(camLayout);
    mainLayout->addWidget(camTableWidget);
    mainLayout->addLayout(dstLayout);
    mainLayout->addWidget(dstTableWidget);

    setLayout(mainLayout);

    connect(loadCamFilePushButton,SIGNAL(clicked()),this,SLOT(loadCamFile()));
    connect(loadDstFilePushButton,SIGNAL(clicked()),this,SLOT(loadDstFile()));



    cameraMatrixLoadedFromFile = false;
    distCoeffsLoadedFromFile = false;


}

UseCam::UseCam(QWidget *parent): QWidget (parent)
{

    caliSingleCam = new caliCam;


    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(caliSingleCam);
    setLayout(mainLayout);


}


UseCam::~UseCam(){
    delete caliSingleCam;
}

SingleCam::SingleCam(QWidget *parent):QDialog(parent)
{
    tabWidget = new QTabWidget;

    useFile = new UseFile;
    useCam = new UseCam;

    tabWidget->addTab(useFile, tr("load parameters from files"));
    tabWidget->addTab(useCam, tr("use source images to calibrate"));



    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
    mainLayout->addWidget(tabWidget);

    setLayout(mainLayout);



}


void UseFile::loadCamFile()
{
    cameraMatrix = cv::Mat::zeros(3,3,CV_64FC1); // double, opencv doc wrong, check code
    cv::FileStorage fs;
    QFileDialog::Options options =  QFileDialog::DontResolveSymlinks;
    QString selectedFilter;
    QMessageBox::information(
                this,
                "file name",
                "the file name should be cam.yaml exactly");
    QString fileName = QFileDialog::getOpenFileName(
                this, tr("open cam.yaml file"),
                "..",
                tr("the file name should be cam.yaml exactly (cam.yaml)"),
                &selectedFilter,
                options);
    if (fileName.isEmpty() || fileName.isNull())        return;
    std::string camMatFileName = fileName.toStdString();


    fs.open(camMatFileName, cv::FileStorage::READ);
    if (!fs.isOpened()){
        QMessageBox::information(this,"file open","camera matrix file open: failed");
        return;
    }
    fs["intrinsic"]>>cameraMatrix;
    Q_ASSERT(cameraMatrix.type()==CV_64FC1); // double, opencv doc wrong, check code
    Q_ASSERT(cameraMatrix.size()==cv::Size(3,3));
    fs.release();

    if (cameraMatrix.empty()){
        QMessageBox::information(this,"file open","no data found in camera matrix");
        return;
    }




    // write pareamter to table
    camTableWidget->clear();
    camTableWidget->setRowCount(3);
    camTableWidget->setColumnCount(3);
    for (int row = 0; row <3; row ++){
        for (int col = 0; col<3; col++){
            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(cameraMatrix.at<double>(row,col)));
            item->setFlags(Qt::NoItemFlags);
            camTableWidget->setItem(row,col,item);
        }
    }

    cameraMatrixLoadedFromFile = true;

}




void UseFile::loadDstFile()
{
    distCoeffs = cv::Mat::zeros(5,1,CV_64FC1); // double, check opencv code
    cv::FileStorage fs;
    std::string dstVecFileName;

    QFileDialog::Options options =  QFileDialog::DontResolveSymlinks;
    QString selectedFilter;
    QMessageBox::information(
                this,
                "file name",
                "the file name should be dst.yaml exactly");
    QString fileName = QFileDialog::getOpenFileName(
                this, tr("open dst.yaml file"),
                "..",
                tr("the file name should be dst.yaml exactly (dst.yaml)"),
                &selectedFilter,
                options);
    if (fileName.isEmpty() || fileName.isNull())        return;
    dstVecFileName = fileName.toStdString();


    fs.open(dstVecFileName, cv::FileStorage::READ);
    if (!fs.isOpened()){
        QMessageBox::information(this,"file open","distortion vector file open: failed");
        return;
    }
    fs["distortion"]>>distCoeffs;
    Q_ASSERT(distCoeffs.size()==cv::Size(1,5)); // .size() returns cols x rows !
    Q_ASSERT(distCoeffs.type()==CV_64FC1); // double, check opencv code
    fs.release();

    if (distCoeffs.empty()){
        QMessageBox::information(this,"file open","no data found in distortion vector");
        return;
    }


    // write pareamter to table
    dstTableWidget->clear();
    dstTableWidget->setRowCount(1);
    dstTableWidget->setColumnCount(5);
    for (int row = 0; row <5; row ++){

            QTableWidgetItem * item = new QTableWidgetItem;

            item->setText(QString::number(distCoeffs.at<double>(row,0)));
            item->setFlags(Qt::NoItemFlags);
            dstTableWidget->setItem(0,row,item);

    }


    distCoeffsLoadedFromFile = true;
}

const cv::Mat & UseFile:: getIntrinsicMatrix(){
    if (!cameraMatrixLoadedFromFile){
        QMessageBox::warning(this,
                             "wrong useage",
                             "camera matrix not yet loaded from file");

    }
    return cameraMatrix;
}

const cv::Mat & UseFile:: getDistortionVector() {

    if (!distCoeffsLoadedFromFile){
        QMessageBox::warning(this,
                             "wrong useage",
                             "distortion vector not yet loaded from file");

    }

    return distCoeffs;
}



