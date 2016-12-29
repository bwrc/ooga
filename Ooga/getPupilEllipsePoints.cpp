#include "getPupilEllipsePoints.h"

#define NORM2(P1, P2) (sqrt((P1.x-P2.x)*(P1.x-P2.x) + (P1.y-P2.y)*(P1.y-P2.y)))

//cv::Point2d* getPupilEllipsePoints(cv::RotatedRect pupilEllipse, cv::Point2d pupilEllipsePoints_prev[4], double theta) {
void getPupilEllipsePoints(cv::RotatedRect pupilEllipse, cv::Point2d pupilEllipsePoints_prev[4], double theta, cv::Point2d* pupilEllipsePoints) {

  // Compute the end points of major and minor axes of the ellipse,returned by fitEllipse function.
  // The first two points will be the major axis points, second two the minor axis points.
  // PupilEllipse has zero angle with the box in 'vertical' orientation, with angles growing anti-clockwise (i.e., the "normal" angle convention).
  // Also, size.height should be the larger dimension, i.e., size.height > size.width should hold when returned by fitEllipse.

  cv::Point2d pupilEllipsePoints_meas[4];

  pupilEllipsePoints_meas[0].x = double(pupilEllipse.center.x + cos(pupilEllipse.angle) * pupilEllipse.size.height/2);
  pupilEllipsePoints_meas[0].y = double(pupilEllipse.center.y - sin(pupilEllipse.angle) * pupilEllipse.size.height/2);
  pupilEllipsePoints_meas[1].x = double(pupilEllipse.center.x - cos(pupilEllipse.angle) * pupilEllipse.size.height/2);
  pupilEllipsePoints_meas[1].y = double(pupilEllipse.center.y + sin(pupilEllipse.angle) * pupilEllipse.size.height/2);
  pupilEllipsePoints_meas[2].x = double(pupilEllipse.center.x + sin(pupilEllipse.angle) * pupilEllipse.size.width/2);
  pupilEllipsePoints_meas[2].y = double(pupilEllipse.center.y + cos(pupilEllipse.angle) * pupilEllipse.size.width/2);
  pupilEllipsePoints_meas[3].x = double(pupilEllipse.center.x - sin(pupilEllipse.angle) * pupilEllipse.size.width/2);
  pupilEllipsePoints_meas[3].y = double(pupilEllipse.center.y - cos(pupilEllipse.angle) * pupilEllipse.size.width/2);

  for (int ind=0; ind<4; ind=ind+2) {
    if (NORM2(pupilEllipsePoints_meas[ind] , pupilEllipsePoints_prev[ind]) > NORM2(pupilEllipsePoints_meas[ind] , pupilEllipsePoints_prev[ind+1])) {
      cv::Point2d tmp = pupilEllipsePoints_meas[ind];
      pupilEllipsePoints_meas[ind] = pupilEllipsePoints_meas[ind+1];
      pupilEllipsePoints_meas[ind+1] = tmp;
      //std::cout << "SWAP! " << std::endl;
    }
  }

  theta = cv::max(theta, 0.2);  // Threshold theta, just to have at least some filtering...

  for (int i=0; i<4; i++)  {
    pupilEllipsePoints[i] = cv::Point2d(theta * pupilEllipsePoints_meas[i].x + (1-theta) * pupilEllipsePoints_prev[i].x, \
					theta * pupilEllipsePoints_meas[i].y + (1-theta) * pupilEllipsePoints_prev[i].y);
  }


}

