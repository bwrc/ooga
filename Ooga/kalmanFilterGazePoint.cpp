#include "kalmanFilterGazePoint.h"

using namespace std;

void kalmanFilterGazePoint(cv::Point2d const loc_meas, cv::Point2d const velo_meas, cv::Mat *param_est, cv::Mat *P_est, double loc_variance) {
  ///    %%% --- Kalman filter implementation with location and velocity as parameters (4D) ---

  double delta = 300; // If the measured point is outside the image more than delta, it is regarded as a bullshit measurement
  double minloc_x = -delta;
  double minloc_y = -delta;
  double maxloc_x = 640+delta;
  double maxloc_y = 480+delta;
  double maxvelo_x = 100;  // If the measured velocity is more than this, it is regarded as a bullshit measurement
  double maxvelo_y = 100;


  /// These constants should be input:
  cv::Mat const Q_const = cv::Mat::eye(4,4, CV_64F);  // the constant covariance of the process noise
  double const Rv_const = 1;      // the constant variance of the velocity measurement noise
  cv::Mat const F = (cv::Mat_<double>(4,4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

  //double R_max = kalman_R_max; // Maximum variance for the observation noise. The larger this is, the less the observations are trusted. (was 100)
  double R12 = loc_variance;
  double const R34 = Rv_const;   // this is kept constant

  //R12 = R_max * (1-theta); // Let this be simpler than in the Kalman filter paper (BSPC) (note that the weighting scheme is in 'theta' now) (INPUT)
  
  // --- PREDICT:
  cv::Mat param_pred = F * (*param_est);
  cv::Mat P_pred = F * (*P_est) * F.t() + Q_const; 

  // --- UPDATE:

  // The measurement is "reasonable":
  if (loc_meas.x > minloc_x && loc_meas.y > minloc_y && loc_meas.x < maxloc_x && loc_meas.y < maxloc_y && abs(velo_meas.x) < maxvelo_x && abs(velo_meas.y) < maxvelo_y) {
    cv::Mat measurement = (cv::Mat_<double>(4,1) << loc_meas.x, loc_meas.y, velo_meas.x, velo_meas.y);  
    cv::Mat resid = measurement - param_pred;
    cv::Mat R_est = cv::Mat::diag((cv::Mat_<double>(4,1) << R12, R12, R34, R34));
    cv::Mat Kgain = P_pred * (P_pred + R_est).inv(cv::DECOMP_SVD);
    *param_est = param_pred + Kgain * resid;
    *P_est = (cv::Mat::eye(4,4, CV_64F) - Kgain) * P_pred;

    
  } else 
    { // The measurement is obviously bullshit --> use the prediction for location and set measurement to zero:
    (*param_est).at<double>(0) = param_pred.at<double>(0);
    (*param_est).at<double>(1) = param_pred.at<double>(1);
    (*param_est).at<double>(2) = 0;
    (*param_est).at<double>(3) = 0;
  }


}
