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

  if (1) {      // Swap axis end points (Yes, do it)
    for (int ind=0; ind<4; ind=ind+2) {
      if (NORM2(pupilEllipsePoints_meas[ind] , pupilEllipsePoints_prev[ind]) > NORM2(pupilEllipsePoints_meas[ind] , pupilEllipsePoints_prev[ind+1])) {
	cv::Point2d tmp = pupilEllipsePoints_meas[ind];
	pupilEllipsePoints_meas[ind] = pupilEllipsePoints_meas[ind+1];
	pupilEllipsePoints_meas[ind+1] = tmp;
	//std::cout << "SWAP! " << std::endl;
      }
    }
  }

  theta = cv::max(theta, 0.2);  // Replace the troublesome filtering in the end of this function with this simple threshold for theta (Miika 22.9.2016)

  for (int i=0; i<4; i++)  {
    pupilEllipsePoints[i] = cv::Point2d(theta * pupilEllipsePoints_meas[i].x + (1-theta) * pupilEllipsePoints_prev[i].x, \
					theta * pupilEllipsePoints_meas[i].y + (1-theta) * pupilEllipsePoints_prev[i].y);
  }

  // if (1) {  // Skip the additional filtering and return here? (It's unnecessary now with the thresholded theta; Miika 22.9.2016)
  //   return; }

  /////////////////////////////////////////////////////////////////////////

}


//   double pupil_diff;
//   if (1) {  // This uses the pupil center and axis end point difference (to be preferred, I guess)
//     pupil_diff = abs(NORM2(pupilEllipsePoints_meas[0], pupilEllipsePoints_meas[1]) - NORM2(pupilEllipsePoints[0], pupilEllipsePoints[1])) + \
//                  abs(NORM2(pupilEllipsePoints_meas[2], pupilEllipsePoints_meas[3]) - NORM2(pupilEllipsePoints[2], pupilEllipsePoints[3])) + \
//       (pupilEllipsePoints_prev[0].x + pupilEllipsePoints_prev[1].x + pupilEllipsePoints_prev[2].x + pupilEllipsePoints_prev[3].x)/4 - \
//       (pupilEllipsePoints[0].x + pupilEllipsePoints[1].x + pupilEllipsePoints[2].x + pupilEllipsePoints[3].x)/4 + \
//       (pupilEllipsePoints_prev[0].y + pupilEllipsePoints_prev[1].y + pupilEllipsePoints_prev[2].y + pupilEllipsePoints_prev[3].y)/4 - \
//       (pupilEllipsePoints[0].y + pupilEllipsePoints[1].y + pupilEllipsePoints[2].y + pupilEllipsePoints[3].y)/4;
    
//   } else {  // This uses only the pupil center difference
//     pupil_diff = \
//       (pupilEllipsePoints_prev[0].x + pupilEllipsePoints_prev[1].x + pupilEllipsePoints_prev[2].x + pupilEllipsePoints_prev[3].x)/4 - \
//       (pupilEllipsePoints[0].x + pupilEllipsePoints[1].x + pupilEllipsePoints[2].x + pupilEllipsePoints[3].x)/4 + \
//       (pupilEllipsePoints_prev[0].y + pupilEllipsePoints_prev[1].y + pupilEllipsePoints_prev[2].y + pupilEllipsePoints_prev[3].y)/4 - \
//       (pupilEllipsePoints[0].y + pupilEllipsePoints[1].y + pupilEllipsePoints[2].y + pupilEllipsePoints[3].y)/4;
//   }

//   /// why such parameters? Often, pupil_diff ~= 0 --> gamma ~= 0.3775
//   //double gamma = 1.0 / (1 + exp(-1 * (pupil_diff - 5)));  // gamma = 1 --> rely only on observations, gamma = 0 --> filter heavily (depending on theta)
//   double gamma = 1.0 / (1 + exp(-1 * (pupil_diff - 0.5)));  // gamma = 1 --> rely only on observations, gamma = 0 --> filter heavily (depending on theta) (was 0.5)
  
//   //std::cout << pupil_diff << " ----> " << gamma << std::endl;
//   //gamma = 0;  // We have Kalman filter now so don't desperately filter here?
  
//   // gamma = 0.3;  // We could as well insert this as we must have gamma>0 in order for this filtering to work.

//   for (int i=0; i<4; i++)  {
//     pupilEllipsePoints[i].x = gamma * pupilEllipsePoints_meas[i].x + (1-gamma) * pupilEllipsePoints[i].x;
//     pupilEllipsePoints[i].y = gamma * pupilEllipsePoints_meas[i].y + (1-gamma) * pupilEllipsePoints[i].y;
//     pupilEllipsePoints_prev[i] = pupilEllipsePoints[i]; // store the previous points here
//   }


//   return pupilEllipsePoints;
// }
