//#include <opencv2/opencv.hpp>

#include "computePupilCenter3d.h"

cv::Point3d computePupilCenter3d(std::vector<cv::Point3d> pupilEllipsePoints3d, cv::Point3d Cc3d) {

  // Constants
  double R = 0.0077;        // corneal radius
  double n_cornea = 1.336;  // defrection index of cornea
  double r_d = 0.0035;      // distance between cornea center and pupil plane, in meters
 
  // Normalize the 3D vectors and denote them with 'K'
  std::vector<cv::Point3d> K_all;
  double inv_norm;
  for (int i=0; i<pupilEllipsePoints3d.size(); i++) {
    inv_norm = 1 / norm(pupilEllipsePoints3d[i]);
    K_all.push_back(pupilEllipsePoints3d[i] * inv_norm);
  }

  std::vector<cv::Point3d> u;
  cv::Point3d K;
  for (int i=0; i<K_all.size(); i++) {
    K = K_all[i];

    // the coefficients of the 2nd degree equation
    double a = 1; //  (K is normalized ---> a=1)
    double b = -2 * Cc3d.dot(K);
    double c = norm(Cc3d)*norm(Cc3d) - R*R;

    double discriminant = b*b - 4*a*c;
    if (discriminant<0) {discriminant = 0; }
    double s = (-b - sqrt(discriminant)) / (2*a);
    u.push_back(s*K);
    

  }

  // The radius of the pupil as half of the major axis (first two K vectors point to the major axis end points, as agreed):
  double r_p = norm(u[0]-u[1]) * norm(u[0]-u[1]) / 2;
  double r_ps2 = r_d*r_d + r_p*r_p;

  cv::Point3d u_hat_sum = cv::Point3d(0,0,0);
  for (int i=0; i<K_all.size(); i++) {
    K = K_all[i];

    // Compute the defracted vector (K_hat) with the vector form of Snell's law
    inv_norm = 1 / norm(u[i] - Cc3d);
    cv::Point3d N_u = (u[i] - Cc3d) * inv_norm;
    cv::Point3d K_hat_unnorm = 1/n_cornea*K + (1/n_cornea*(-N_u.dot(K)) - sqrt(1 - 1/(n_cornea*n_cornea)*(1-(N_u.dot(K)*N_u.dot(K)))))*N_u;
    inv_norm = 1 / norm(K_hat_unnorm);
    cv::Point3d K_hat = K_hat_unnorm * inv_norm;   // normalized vector from u_i to u_i'

    // Compute the 'w' in u_hat = u + w*K_hat
    double a = 1; //  (K_hat is normalized ---> a=1)
    double b = 2*(u[i].dot(K_hat) - Cc3d.dot(K_hat));
    double c = norm(u[i])*norm(u[i]) - 2*Cc3d.dot(u[i]) + norm(Cc3d)*norm(Cc3d) - r_ps2;
    double discriminant = b*b - 4*a*c;
    if (discriminant<0) {discriminant = 0; }
    double w = (-b - sqrt(discriminant)) / (2*a);
    u_hat_sum = u_hat_sum + u[i] + w*K_hat;

  }

  double inv_N = 1 / double(K_all.size()); 
  return u_hat_sum * inv_N;

}
