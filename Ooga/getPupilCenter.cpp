//#include <opencv2/opencv.hpp>

#include "getPupilCenter.h"

cv::Point2d getPupilCenter(cv::Mat eyeImage_filtered) {

  // Initialize variables
  float Svert = eyeImage_filtered.rows;
  float Shori = eyeImage_filtered.cols;

  cv::Mat eyeImage_cropped;
  double min_val, max_val;
  cv::Point min_loc, max_loc;
  cv::Point2d pupil_center_appro;

  int delta = 20;  // Crop the edges away (delta must be smaller than half of the image size)
  cv::Mat eyeImage_filtered_cropped = eyeImage_filtered(cv::Range(delta,Svert-delta) , cv::Range(delta,Shori-delta));

  minMaxLoc(eyeImage_filtered_cropped, &min_val, &max_val, &min_loc, &max_loc);
  pupil_center_appro.x = min_loc.x + delta;
  pupil_center_appro.y = min_loc.y + delta;

  return pupil_center_appro;
}
