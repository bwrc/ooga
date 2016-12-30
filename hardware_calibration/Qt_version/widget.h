#ifndef WIDGET_H
#define WIDGET_H


#include <QDialog>

#include "pages.h"

QT_BEGIN_NAMESPACE
class QListWidget;
class QListWidgetItem;
class QStackedWidget;
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT
    
public:
    Widget();
    void caliPageActivating();
    
public slots:
    void changePage(QListWidgetItem *current, QListWidgetItem *previous);
    void calibrate();
    void saveResult(const cv::Mat & x);

private:


    void createIcons();
    QListWidget * iconListWidget;
    QStackedWidget * pageStackedWidget;


    EyeCamPage * eyeCamPage;
    SceneCamPage * sceneCamPage;
    RigPage * rigPage;
    CaptureImagePairsPage * captureImagePairsPage;
    CaliPage * caliPage;



};

#endif // WIDGET_H
