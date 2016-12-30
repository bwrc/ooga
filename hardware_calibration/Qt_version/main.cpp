

#include <QApplication>
#include "widget.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{


    cv::Mat testi;

  //Q_INIT_RESOURCE(application_resource);

  QApplication a(argc, argv);





  Widget w;
  w.show();
    
  return a.exec();

}
