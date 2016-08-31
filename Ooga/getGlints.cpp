//#include <opencv2/opencv.hpp>

//const double PI = 3.1415926535898;

#include "getGlints.h"
#include "PerformanceTimer.h"

using namespace cv;
using namespace std;

Mat log_mvnpdf(Mat x, Mat mu, Mat C) {
  // Computes the logarithm of multivariate normal probability density function (mvnpdf) of x with 
  // mean mu, covariance C, and log'd normalization factor log_norm_factor, using Cholesky decomposition.
  // The matrix sizes must be [D x N] where D is the dimension and N is the number of points in which to evalute this function.
  // Give the arguments as CV_32F.

  float log_norm_factor = -0.5*(2*log(2*PI) + log(determinant(C)));
  Mat invChol = C.clone();
  Cholesky(invChol.ptr<float>(), invChol.step, invChol.cols, 0, 0, 0);
  // Opencv's 'Cholesky' returns a bit weard matrix... it's almost the inverse of real cholesky but the off-diagonals must be adjusted:
  invChol.at<float>(0,1) = -invChol.at<float>(0,0)*invChol.at<float>(1,1)*invChol.at<float>(1,0);
  invChol.at<float>(1,0) = 0;

  Mat x_mu(mu.rows, mu.cols, CV_32F);
  x_mu = x - mu;

  Mat mahal2(x.cols, x.rows, CV_32F);
  Mat mahal1(x.cols, x.rows, CV_32F);
  mahal1 = x_mu.t() * invChol;
  pow(mahal1, 2, mahal2);
  Mat mahal2_sum(x.cols, 1, CV_32F);
  reduce(mahal2, mahal2_sum, 1, CV_REDUCE_SUM);

  return -0.5*mahal2_sum.t() + log_norm_factor;

}


std::vector<Point2d> getGlints(Mat eyeImage_diff, Point2d pupil_center, std::vector<cv::Point2d> glintPoints_prev, float theta, float MU_X[], float MU_Y[], Mat CM, cv::Mat glint_kernel, double *score) {


  // Set parameters:
  float Svert = eyeImage_diff.rows;   // Image size, vertical
  float Shori = eyeImage_diff.cols;   // Image size, horizontal
  const int N_leds = 6;     // Number of leds
  /*KL next line: this needs to be constant to work in array init on ms compilers*/
  const int N_glint_candidates = 6;   // Number of glint candidates (It would be nice to have a largish number here, such as 6. However, proctime = O(N_glint_candidates) )
  float delta_x = 50;  // zoom area (horizontal) where to search the glints
  float delta_y = 50;  // zoom area (vertical) where to search the glints
  int delta_glint = 10;  // In searching for glint candidates, insert this amount of zero pixels around each found candidate
  float beta = 50;       // the likelihood parameter
  float reg_coef = 3;   // regularization coefficient for the covariance matrix (was 5)
  float delta_extra = 5;   // Some extra for cropping the area where the likelihood will be evaluated (was 10)

  // Initialize variables
  int N_particles = N_leds * N_glint_candidates;
  Mat eyeImage_filtered(Svert, Shori, CV_8UC1);
  Mat eyeImage_filtered_clone(Svert, Shori, CV_8UC1);
  Mat eyeImage_aux(Svert, Shori, CV_8UC1);
  Mat eyeImage_cropped;
  Mat eyeImage_aux_crop;
  float glint_candidates[2][N_glint_candidates];
  float glints_x[N_glint_candidates*N_leds][N_leds];
  float glints_y[N_glint_candidates*N_leds][N_leds];
  int ORDERS[N_leds][N_leds];
  double min_val, max_val;
  Point min_loc, max_loc;
  float im_max;
  float x_prev;
  float y_prev;
  Mat mu_cond(2,1, CV_32F);
  Mat mu_dyn(2,1, CV_32F);
  Mat mu_dum(2,1, CV_32F);
  Mat C_cond(2, 2, CV_32F);
  Mat C_dyn(2,2, CV_32F); C_dyn = Mat::eye(2,2, CV_32F)*theta*100; // (loppukerroin oli 100)
  Mat invC_dyn(2,2, CV_32F);
  if (invert(C_dyn, invC_dyn, DECOMP_CHOLESKY) == 0) {    printf("C_dynamical could not be inverted with theta = %.100f \n", theta);  exit(-1);     }
  int x_min;
  int x_max;
  int y_min;
  int y_max;
  float largest_particle_weight = FLT_MIN;
  int best_particle = 0;
  double log_prior_scaling_factor = 0;

  std::vector< std::pair<double, std::string>> timestamps;

  TPerformanceTimer* pt = new TPerformanceTimer();
  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "start glints"));

  // Filter the image
  filter2D(eyeImage_diff, eyeImage_filtered, -1, glint_kernel, Point(-1,-1), 0, BORDER_REPLICATE );
  minMaxLoc(eyeImage_filtered, &min_val, &max_val, &min_loc, &max_loc);
  im_max = float(max_val);

  eyeImage_filtered_clone = eyeImage_filtered.clone();

  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "filtered"));

  // Crop the image around the pupil
  // NOTE: In opencv, Range(a,b) is the same as a:b-1 in Matlab, i.e., an inclusive left boundary and an exclusive right boundary, i.e., [a,b) !
  eyeImage_cropped = eyeImage_filtered(Range(max(float(0), float(pupil_center.y)-delta_y), min(Svert, float(pupil_center.y)+delta_y)) , \
				       Range(max(float(0), float(pupil_center.x)-delta_x), min(Shori, float(pupil_center.x)+delta_x)));

  float Svert_crop = eyeImage_cropped.rows;
  float Shori_crop = eyeImage_cropped.cols;

  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "localize glints:"));

  for (int i=0; i<N_glint_candidates; i++) {
    minMaxLoc(eyeImage_cropped, &min_val, &max_val, &min_loc, &max_loc);
    glint_candidates[0][i] = max_loc.x + float(pupil_center.x)-(delta_x+1);
    glint_candidates[1][i] = max_loc.y + float(pupil_center.y)-(delta_y+1);
    eyeImage_cropped(Range(max(float(0), float(max_loc.y-delta_glint)) , min(Svert_crop, float(max_loc.y+delta_glint))) , \
		     Range(max(float(0), float(max_loc.x-delta_glint)) , min(Shori_crop, float(max_loc.x+delta_glint))) ) = \
      eyeImage_cropped(Range(max(float(0), float(max_loc.y-delta_glint)) , min(Svert_crop, float(max_loc.y+delta_glint))) , \
    		       Range(max(float(0), float(max_loc.x-delta_glint)) , min(Shori_crop, float(max_loc.x+delta_glint))) ) * 0;
  }

  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "k:"));


  int k=0;
  for (int j=0; j<N_glint_candidates; j++) {
    for (int n=0; n<N_leds; n++) {
      for (int m=0; m<N_leds; m++) {
	if (m==n) {
	  glints_x[j*N_leds+n][m] = glint_candidates[0][k];
	  glints_y[j*N_leds+n][m] = glint_candidates[1][k];
	}
	else {
	  glints_x[j*N_leds+n][m] = 0;
	  glints_y[j*N_leds+n][m] = 0;
	}
      }
    }
    k++;
  }

  for (int j=0; j<N_leds; j++) {
    for (int n=0; n<N_leds-j; n++) {
      ORDERS[j][n] = n+j; }
    for (int n=N_leds-j; n<N_leds; n++) {
      ORDERS[j][n] = n-(N_leds-j); }
  }

  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "insertzeros"));

  int p = 0;
  for (int n=0; n<N_glint_candidates; n++) {
    for (int j=0; j<N_leds; j++) {
      float log_post_MAP = 0;
      eyeImage_aux = eyeImage_filtered_clone.clone();
      double log_scaling_factor_mean = 0;
      for (int k=1; k<N_leds; k++) {

	// Insert zeros around the already sampled (i.e., previous) glint:
	x_prev = glints_x[p][ORDERS[j][k-1]];
	y_prev = glints_y[p][ORDERS[j][k-1]];
	y_min = max(float(0), min(float(Svert)-1, float(y_prev-delta_glint)));
	y_max = max(float(0), min(float(Svert), float(y_prev+delta_glint)));
	x_min = max(float(0), min(float(Shori)-1, float(x_prev-delta_glint)));
	x_max = max(float(0), min(float(Shori), float(x_prev+delta_glint)));
	eyeImage_aux(Range(y_min , y_max) , Range(x_min , x_max)) = eyeImage_aux(Range(y_min , y_max) , Range(x_min , x_max)) * 0;

	// Set some helper variables
	//KLfloat glints_x_sofar[k], glints_y_sofar[k],  MU_X_sofar[k],  MU_Y_sofar[k], glints_x_sofar_mean = 0, glints_y_sofar_mean = 0, MU_X_sofar_mean = 0, MU_Y_sofar_mean = 0;
	float *glints_x_sofar = new float[k];
	float *glints_y_sofar = new float[k];
	float *MU_X_sofar = new float[k];
	float *MU_Y_sofar = new float[k];
	float glints_x_sofar_mean = 0, glints_y_sofar_mean = 0, MU_X_sofar_mean = 0, MU_Y_sofar_mean = 0;

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "iterate k"));

	for (int i=0; i<k; i++) {
	  glints_x_sofar[i] = glints_x[p][ORDERS[j][i]];
	  glints_y_sofar[i] = glints_y[p][ORDERS[j][i]];
	  glints_x_sofar_mean = glints_x_sofar_mean + glints_x_sofar[i];
	  glints_y_sofar_mean = glints_y_sofar_mean + glints_y_sofar[i];
	  MU_X_sofar[i] = MU_X[ORDERS[j][i]];
	  MU_Y_sofar[i] = MU_Y[ORDERS[j][i]];
	  MU_X_sofar_mean = MU_X_sofar_mean + MU_X[ORDERS[j][i]];
	  MU_Y_sofar_mean = MU_Y_sofar_mean + MU_Y[ORDERS[j][i]];
	}
	glints_x_sofar_mean = glints_x_sofar_mean / k;
	glints_y_sofar_mean = glints_y_sofar_mean / k;
	MU_X_sofar_mean = MU_X_sofar_mean / k;
	MU_Y_sofar_mean = MU_Y_sofar_mean / k;

	// Compute the mean without covariance (= "dummy mean"):
	mu_dum.at<float>(0,0) = glints_x_sofar_mean + MU_X[ORDERS[j][k]] - MU_X_sofar_mean;
	mu_dum.at<float>(1,0) = glints_y_sofar_mean + MU_Y[ORDERS[j][k]] - MU_Y_sofar_mean;
	  	  
	// Compute the conditional error:
	Mat error_cond(2*k, 1, CV_32F);
	for (int i=0; i<k; i++) {
	  error_cond.at<float>(i,0) = glints_x_sofar[i] - glints_x_sofar_mean - (MU_X_sofar[i] - MU_X_sofar_mean);
	  error_cond.at<float>(k+i,0) = glints_y_sofar[i] - glints_y_sofar_mean - (MU_Y_sofar[i] - MU_Y_sofar_mean);
	}

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "comp CM"));

	// Compute the four blocks of the CM matrix
	Mat CM_topLeft(2, 2, CV_32F);
	CM_topLeft.at<float>(0,0) = CM.at<float>(ORDERS[j][k] , ORDERS[j][k]);
	CM_topLeft.at<float>(0,1) = CM.at<float>(ORDERS[j][k] , ORDERS[j][k]+N_leds);
	CM_topLeft.at<float>(1,0) = CM.at<float>(ORDERS[j][k]+N_leds , ORDERS[j][k]);
	CM_topLeft.at<float>(1,1) = CM.at<float>(ORDERS[j][k]+N_leds , ORDERS[j][k]+N_leds);

	Mat CM_topRight(2, k*2, CV_32F);
	for (int i=0; i<k; i++)  {
	  CM_topRight.at<float>(0,i) = CM.at<float>(ORDERS[j][k] , ORDERS[j][i]);
	  CM_topRight.at<float>(0,k+i) = CM.at<float>(ORDERS[j][k] , ORDERS[j][i]+N_leds);
	  CM_topRight.at<float>(1,i) = CM.at<float>(ORDERS[j][k]+N_leds , ORDERS[j][i]);
	  CM_topRight.at<float>(1,k+i) = CM.at<float>(ORDERS[j][k]+N_leds , ORDERS[j][i]+N_leds);
	}

	Mat CM_bottomRight(k*2, k*2, CV_32F);
	for (int i1=0; i1<k; i1++)  {
	  for (int i2=0; i2<k; i2++)  {
	    CM_bottomRight.at<float>(i1,i2) = CM.at<float>(ORDERS[j][i1] , ORDERS[j][i2]);
	    CM_bottomRight.at<float>(i1,k+i2) = CM.at<float>(ORDERS[j][i1] , ORDERS[j][i2]+N_leds);
	    CM_bottomRight.at<float>(k+i1,i2) = CM.at<float>(ORDERS[j][i1]+N_leds , ORDERS[j][i2]);
	    CM_bottomRight.at<float>(k+i1,k+i2) = CM.at<float>(ORDERS[j][i1]+N_leds , ORDERS[j][i2]+N_leds);
	  }
	}

	Mat invCM_bottomRight(k*2, k*2, CV_32F);
	if (invert(CM_bottomRight, invCM_bottomRight, DECOMP_CHOLESKY) == 0) {
	  printf("The bottom right block of CM could not be inverted [n=%d, j=%d, k=%d] \n", n,j,k); exit(-1); }

	Mat CM_bottomLeft(k*2, k*2, CV_32F);
	CM_bottomLeft = CM_topRight.t();  // Covariance matrix is always symmetric

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "cov-corrected"));

	// Compute the covariance-corrected conditional mean
	mu_cond = mu_dum + CM_topRight * invCM_bottomRight * error_cond;	  
	if (mu_cond.at<float>(0,0) > Shori-1) {mu_cond.at<float>(0,0) = Shori; }   // Let's force mu inside the image, if it's not
	if (mu_cond.at<float>(1,0) > Svert-1) {mu_cond.at<float>(1,0) = Svert; }

	// Compute the covariance-corrected conditional covariance
	C_cond = CM_topLeft - CM_topRight * invCM_bottomRight * CM_bottomLeft;

	// Regularize the covariance
	C_cond.at<float>(0,0) = C_cond.at<float>(0,0) + reg_coef;
	C_cond.at<float>(1,1) = C_cond.at<float>(1,1) + reg_coef;

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "likelihood"));

	// Form the likelihood (define the crop area first) (PITÄISKÖ MÄÄRITTÄÄ ALUE ENEMMÄNKIN C_COMB:N PERUSTEELLA ?)
	x_min = max(0, min(int(Shori)-1, int(round(mu_cond.at<float>(0,0) - 3*C_cond.at<float>(0,0) - delta_extra))));
	x_max = max(1, min(int(Shori), int(round(mu_cond.at<float>(0,0) + 3*C_cond.at<float>(0,0) + delta_extra))));
	y_min = max(0, min(int(Svert)-1, int(round(mu_cond.at<float>(1,0) - 3*C_cond.at<float>(1,1) - delta_extra))));
	y_max = max(1, min(int(Svert), int(round(mu_cond.at<float>(1,0) + 3*C_cond.at<float>(1,1) + delta_extra))));
	eyeImage_aux_crop = eyeImage_aux(Range(y_min, y_max), Range(x_min, x_max));
	Mat eyeImage_aux_crop_float(eyeImage_aux_crop.rows, eyeImage_aux_crop.cols, CV_32F);
	eyeImage_aux_crop.convertTo(eyeImage_aux_crop_float, CV_32F);
	Mat log_lhood(y_max-y_min, x_max-x_min, CV_32F);
	log_lhood = beta * eyeImage_aux_crop_float / im_max;

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "cond prior"));

	// Form the conditional prior distribution
	Mat log_prior(y_max-y_min, x_max-x_min, CV_32F);
	Mat xv(1, x_max-x_min, CV_32F);   // The coordinates for which
	Mat yv(y_max-y_min, 1, CV_32F);   //  to evaluate the mvnpdf
	int ii=0; for (int x=x_min; x<x_max; x++) {xv.at<float>(0,ii++) = x;}
	ii=0;     for (int y=y_min; y<y_max; y++) {yv.at<float>(ii++,0) = y;}
	Mat xx = repeat(xv, yv.total(), 1);
	Mat yy = repeat(yv, 1, xv.total());
	Mat coords(2, xx.total(), CV_32F);
	Mat xCol(coords(Range(0,1), Range::all()));
	(xx.reshape(0,1)).copyTo(xCol);
	Mat yCol(coords(Range(1,2), Range::all()));
	(yy.reshape(0,1)).copyTo(yCol);
	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "theta-else"));

	if (theta < 0)  {  // Use only the conditional prior
	  log_prior = log_mvnpdf(coords, repeat(mu_cond, 1, coords.cols), C_cond).reshape(0, xx.rows);

	} else {  // Combine the conditional and dynamical prior
	  // Form the product of two mvnpdf's; see, e.g., http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf

	  mu_dyn.at<float>(0,0) = float(glintPoints_prev[ORDERS[j][k]].x);
	  mu_dyn.at<float>(1,0) = float(glintPoints_prev[ORDERS[j][k]].y);
	  Mat C_sum = C_cond + C_dyn;
	  Mat log_scaling_factor = log_mvnpdf(mu_cond, mu_dyn, C_sum);  // This is actually a scalar
	  Mat invC_cond(2, 2, CV_32F);
	  if (invert(C_cond, invC_cond, DECOMP_CHOLESKY) == 0) { printf("C_conditional could not be inverted \n");  exit(-1);     }
	  Mat C_comb(2, 2, CV_32F);
	  if (invert(invC_cond + invC_dyn, C_comb, DECOMP_CHOLESKY) == 0) { printf("C_combined could not be inverted \n");  exit(-1);     }
	  log_prior = log_scaling_factor.at<float>(0,0) + log_mvnpdf(coords , repeat(C_comb * (invC_cond*mu_cond + invC_dyn*mu_dyn), 1, coords.cols) , C_comb).reshape(0, xx.rows);

	  log_scaling_factor_mean = log_scaling_factor_mean + 1.0/N_leds*double(log_scaling_factor.at<float>(0,0));
	}

	timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "log posterior"));

	// Compute the logarithm of posterior distribution
	Mat log_post(y_max-y_min, x_max-x_min, CV_32F);
	log_post = log_prior + log_lhood;

	// The glint location shall be the MAP estimate
	minMaxLoc(log_post, &min_val, &max_val, &min_loc, &max_loc);
	glints_x[p][ORDERS[j][k]] = max_loc.x + x_min;
	glints_y[p][ORDERS[j][k]] = max_loc.y + y_min;
	log_post_MAP = log_post_MAP + max_val;

	//KL delete dynamically reserved arrays
	delete[] glints_x_sofar;
	delete[] glints_y_sofar;
	delete[] MU_X_sofar;
	delete[] MU_Y_sofar;

      }  // end of k loop

	  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "findwinner"));

      // Find the winner particle (i.e., with largest weight)
      if (log_post_MAP > largest_particle_weight) {
	best_particle = p;
	largest_particle_weight = log_post_MAP;
	log_prior_scaling_factor = log_scaling_factor_mean;
      }
      p++;  // increase the index of particle

    }  // end of j loop
  }  // end of n loop

    
  *score = largest_particle_weight / (N_leds * (beta + log_prior_scaling_factor));  // heuristic score for the match

  // Return the particle with the largest weight
  Point2d glintPoint;
  std::vector<cv::Point2d> glintPoints;
  for (int i=0; i<N_leds; i++) {
    glintPoint.x = glints_x[best_particle][i];
    glintPoint.y = glints_y[best_particle][i];
    glintPoints.push_back(glintPoint);
  }

  timestamps.push_back(std::pair<double, std::string>(pt->elapsed_ms(), "done"));

  for (auto& item : timestamps) {
	  std::cout << std::get<0>(item) << " - " << std::get<1>(item) << std::endl;
  }

  delete pt;

  return glintPoints;

}


void getGlintsThreaded(std::vector<Point2d> &glintPoints, 
	Mat eyeImage_diff, 
	Point2d pupil_center, 
	std::vector<cv::Point2d> glintPoints_prev, 
	float theta, 
	float MU_X[], 
	float MU_Y[], 
	cv::Mat CM, 
	cv::Mat glint_kernel, 
	double *score) {

	// Set parameters:
	float Svert = eyeImage_diff.rows;   // Image size, vertical
	float Shori = eyeImage_diff.cols;   // Image size, horizontal
	const int N_leds = 6;     // Number of leds
	/*KL next line: this needs to be constant to work in array init on ms compilers*/
	const int N_glint_candidates = 2;   // Number of glint candidates (It would be nice to have a largish number here, such as 6. However, proctime = O(N_glint_candidates) )
	float delta_x = 50;  // zoom area (horizontal) where to search the glints
	float delta_y = 50;  // zoom area (vertical) where to search the glints
	int delta_glint = 10;  // In searching for glint candidates, insert this amount of zero pixels around each found candidate
	float beta = 50;       // the likelihood parameter
	float reg_coef = 3;   // regularization coefficient for the covariance matrix (was 5)
	float delta_extra = 5;   // Some extra for cropping the area where the likelihood will be evaluated (was 10)

	// Initialize variables
	int N_particles = N_leds * N_glint_candidates;
	Mat eyeImage_filtered(Svert, Shori, CV_8UC1);
	Mat eyeImage_filtered_clone(Svert, Shori, CV_8UC1);
	Mat eyeImage_aux(Svert, Shori, CV_8UC1);
	Mat eyeImage_cropped;
	Mat eyeImage_aux_crop;
	float glint_candidates[2][N_glint_candidates];
	float glints_x[N_glint_candidates*N_leds][N_leds];
	float glints_y[N_glint_candidates*N_leds][N_leds];
	int ORDERS[N_leds][N_leds];
	double min_val, max_val;
	Point min_loc, max_loc;
	float im_max;
	float x_prev;
	float y_prev;
	Mat mu_cond(2, 1, CV_32F);
	Mat mu_dyn(2, 1, CV_32F);
	Mat mu_dum(2, 1, CV_32F);
	Mat C_cond(2, 2, CV_32F);
	Mat C_dyn(2, 2, CV_32F); C_dyn = Mat::eye(2, 2, CV_32F)*theta * 100; // (loppukerroin oli 100)
	Mat invC_dyn(2, 2, CV_32F);
	if (invert(C_dyn, invC_dyn, DECOMP_CHOLESKY) == 0) { printf("C_dynamical could not be inverted with theta = %.100f \n", theta);  exit(-1); }
	int x_min;
	int x_max;
	int y_min;
	int y_max;
	float largest_particle_weight = FLT_MIN;
	int best_particle = 0;


	// Filter the image
	filter2D(eyeImage_diff, eyeImage_filtered, -1, glint_kernel, Point(-1, -1), 0, BORDER_REPLICATE);
	minMaxLoc(eyeImage_filtered, &min_val, &max_val, &min_loc, &max_loc);
	im_max = float(max_val);

	eyeImage_filtered_clone = eyeImage_filtered.clone();

	// Crop the image around the pupil
	// NOTE: In opencv, Range(a,b) is the same as a:b-1 in Matlab, i.e., an inclusive left boundary and an exclusive right boundary, i.e., [a,b) !
	eyeImage_cropped = eyeImage_filtered(Range(max(float(0), float(pupil_center.y) - delta_y), min(Svert, float(pupil_center.y) + delta_y)), \
		Range(max(float(0), float(pupil_center.x) - delta_x), min(Shori, float(pupil_center.x) + delta_x)));

	float Svert_crop = eyeImage_cropped.rows;
	float Shori_crop = eyeImage_cropped.cols;

	for (int i = 0; i<N_glint_candidates; i++) {
		minMaxLoc(eyeImage_cropped, &min_val, &max_val, &min_loc, &max_loc);
		glint_candidates[0][i] = max_loc.x + float(pupil_center.x) - (delta_x + 1);
		glint_candidates[1][i] = max_loc.y + float(pupil_center.y) - (delta_y + 1);
		eyeImage_cropped(Range(max(float(0), float(max_loc.y - delta_glint)), min(Svert_crop, float(max_loc.y + delta_glint))), \
			Range(max(float(0), float(max_loc.x - delta_glint)), min(Shori_crop, float(max_loc.x + delta_glint)))) = \
			eyeImage_cropped(Range(max(float(0), float(max_loc.y - delta_glint)), min(Svert_crop, float(max_loc.y + delta_glint))), \
			Range(max(float(0), float(max_loc.x - delta_glint)), min(Shori_crop, float(max_loc.x + delta_glint)))) * 0;
	}

	int k = 0;
	for (int j = 0; j<N_glint_candidates; j++) {
		for (int n = 0; n<N_leds; n++) {
			for (int m = 0; m<N_leds; m++) {
				if (m == n) {
					glints_x[j*N_leds + n][m] = glint_candidates[0][k];
					glints_y[j*N_leds + n][m] = glint_candidates[1][k];
				}
				else {
					glints_x[j*N_leds + n][m] = 0;
					glints_y[j*N_leds + n][m] = 0;
				}
			}
		}
		k++;
	}

	for (int j = 0; j<N_leds; j++) {
		for (int n = 0; n<N_leds - j; n++) {
			ORDERS[j][n] = n + j;
		}
		for (int n = N_leds - j; n<N_leds; n++) {
			ORDERS[j][n] = n - (N_leds - j);
		}
	}


	int p = 0;
	for (int n = 0; n<N_glint_candidates; n++) {
		for (int j = 0; j<N_leds; j++) {
			float log_post_MAP = 0;
			eyeImage_aux = eyeImage_filtered_clone.clone();
			for (int k = 1; k<N_leds; k++) {

				// Insert zeros around the already sampled (i.e., previous) glint:
				x_prev = glints_x[p][ORDERS[j][k - 1]];
				y_prev = glints_y[p][ORDERS[j][k - 1]];
				y_min = max(float(0), min(float(Svert) - 1, float(y_prev - delta_glint)));
				y_max = max(float(0), min(float(Svert), float(y_prev + delta_glint)));
				x_min = max(float(0), min(float(Shori) - 1, float(x_prev - delta_glint)));
				x_max = max(float(0), min(float(Shori), float(x_prev + delta_glint)));
				eyeImage_aux(Range(y_min, y_max), Range(x_min, x_max)) = eyeImage_aux(Range(y_min, y_max), Range(x_min, x_max)) * 0;

				// Set some helper variables
				//KLfloat glints_x_sofar[k], glints_y_sofar[k],  MU_X_sofar[k],  MU_Y_sofar[k], glints_x_sofar_mean = 0, glints_y_sofar_mean = 0, MU_X_sofar_mean = 0, MU_Y_sofar_mean = 0;
				float *glints_x_sofar = new float[k];
				float *glints_y_sofar = new float[k];
				float *MU_X_sofar = new float[k];
				float *MU_Y_sofar = new float[k];
				float glints_x_sofar_mean = 0, glints_y_sofar_mean = 0, MU_X_sofar_mean = 0, MU_Y_sofar_mean = 0;

				for (int i = 0; i<k; i++) {
					glints_x_sofar[i] = glints_x[p][ORDERS[j][i]];
					glints_y_sofar[i] = glints_y[p][ORDERS[j][i]];
					glints_x_sofar_mean = glints_x_sofar_mean + glints_x_sofar[i];
					glints_y_sofar_mean = glints_y_sofar_mean + glints_y_sofar[i];
					MU_X_sofar[i] = MU_X[ORDERS[j][i]];
					MU_Y_sofar[i] = MU_Y[ORDERS[j][i]];
					MU_X_sofar_mean = MU_X_sofar_mean + MU_X[ORDERS[j][i]];
					MU_Y_sofar_mean = MU_Y_sofar_mean + MU_Y[ORDERS[j][i]];
				}
				glints_x_sofar_mean = glints_x_sofar_mean / k;
				glints_y_sofar_mean = glints_y_sofar_mean / k;
				MU_X_sofar_mean = MU_X_sofar_mean / k;
				MU_Y_sofar_mean = MU_Y_sofar_mean / k;

				// Compute the mean without covariance (= "dummy mean"):
				mu_dum.at<float>(0, 0) = glints_x_sofar_mean + MU_X[ORDERS[j][k]] - MU_X_sofar_mean;
				mu_dum.at<float>(1, 0) = glints_y_sofar_mean + MU_Y[ORDERS[j][k]] - MU_Y_sofar_mean;

				// Compute the conditional error:
				Mat error_cond(2 * k, 1, CV_32F);
				for (int i = 0; i<k; i++) {
					error_cond.at<float>(i, 0) = glints_x_sofar[i] - glints_x_sofar_mean - (MU_X_sofar[i] - MU_X_sofar_mean);
					error_cond.at<float>(k + i, 0) = glints_y_sofar[i] - glints_y_sofar_mean - (MU_Y_sofar[i] - MU_Y_sofar_mean);
				}

				// Compute the four blocks of the CM matrix
				Mat CM_topLeft(2, 2, CV_32F);
				CM_topLeft.at<float>(0, 0) = CM.at<float>(ORDERS[j][k], ORDERS[j][k]);
				CM_topLeft.at<float>(0, 1) = CM.at<float>(ORDERS[j][k], ORDERS[j][k] + N_leds);
				CM_topLeft.at<float>(1, 0) = CM.at<float>(ORDERS[j][k] + N_leds, ORDERS[j][k]);
				CM_topLeft.at<float>(1, 1) = CM.at<float>(ORDERS[j][k] + N_leds, ORDERS[j][k] + N_leds);

				Mat CM_topRight(2, k * 2, CV_32F);
				for (int i = 0; i<k; i++)  {
					CM_topRight.at<float>(0, i) = CM.at<float>(ORDERS[j][k], ORDERS[j][i]);
					CM_topRight.at<float>(0, k + i) = CM.at<float>(ORDERS[j][k], ORDERS[j][i] + N_leds);
					CM_topRight.at<float>(1, i) = CM.at<float>(ORDERS[j][k] + N_leds, ORDERS[j][i]);
					CM_topRight.at<float>(1, k + i) = CM.at<float>(ORDERS[j][k] + N_leds, ORDERS[j][i] + N_leds);
				}

				Mat CM_bottomRight(k * 2, k * 2, CV_32F);
				for (int i1 = 0; i1<k; i1++)  {
					for (int i2 = 0; i2<k; i2++)  {
						CM_bottomRight.at<float>(i1, i2) = CM.at<float>(ORDERS[j][i1], ORDERS[j][i2]);
						CM_bottomRight.at<float>(i1, k + i2) = CM.at<float>(ORDERS[j][i1], ORDERS[j][i2] + N_leds);
						CM_bottomRight.at<float>(k + i1, i2) = CM.at<float>(ORDERS[j][i1] + N_leds, ORDERS[j][i2]);
						CM_bottomRight.at<float>(k + i1, k + i2) = CM.at<float>(ORDERS[j][i1] + N_leds, ORDERS[j][i2] + N_leds);
					}
				}

				Mat invCM_bottomRight(k * 2, k * 2, CV_32F);
				if (invert(CM_bottomRight, invCM_bottomRight, DECOMP_CHOLESKY) == 0) {
					printf("The bottom right block of CM could not be inverted [n=%d, j=%d, k=%d] \n", n, j, k); exit(-1);
				}

				Mat CM_bottomLeft(k * 2, k * 2, CV_32F);
				CM_bottomLeft = CM_topRight.t();  // Covariance matrix is always symmetric


				// Compute the covariance-corrected conditional mean
				mu_cond = mu_dum + CM_topRight * invCM_bottomRight * error_cond;
				if (mu_cond.at<float>(0, 0) > Shori - 1) { mu_cond.at<float>(0, 0) = Shori; }   // Let's force mu inside the image, if it's not
				if (mu_cond.at<float>(1, 0) > Svert - 1) { mu_cond.at<float>(1, 0) = Svert; }

				// Compute the covariance-corrected conditional covariance
				C_cond = CM_topLeft - CM_topRight * invCM_bottomRight * CM_bottomLeft;

				// Regularize the covariance
				C_cond.at<float>(0, 0) = C_cond.at<float>(0, 0) + reg_coef;
				C_cond.at<float>(1, 1) = C_cond.at<float>(1, 1) + reg_coef;

				// Form the likelihood (define the crop area first) (PITÄISKÖ MÄÄRITTÄÄ ALUE ENEMMÄNKIN C_COMB:N PERUSTEELLA ?)
				x_min = max(0, min(int(Shori) - 1, int(round(mu_cond.at<float>(0, 0) - 3 * C_cond.at<float>(0, 0) - delta_extra))));
				x_max = max(1, min(int(Shori), int(round(mu_cond.at<float>(0, 0) + 3 * C_cond.at<float>(0, 0) + delta_extra))));
				y_min = max(0, min(int(Svert) - 1, int(round(mu_cond.at<float>(1, 0) - 3 * C_cond.at<float>(1, 1) - delta_extra))));
				y_max = max(1, min(int(Svert), int(round(mu_cond.at<float>(1, 0) + 3 * C_cond.at<float>(1, 1) + delta_extra))));
				eyeImage_aux_crop = eyeImage_aux(Range(y_min, y_max), Range(x_min, x_max));
				Mat eyeImage_aux_crop_float(eyeImage_aux_crop.rows, eyeImage_aux_crop.cols, CV_32F);
				eyeImage_aux_crop.convertTo(eyeImage_aux_crop_float, CV_32F);
				Mat log_lhood(y_max - y_min, x_max - x_min, CV_32F);
				log_lhood = beta * eyeImage_aux_crop_float / im_max;

				// Form the conditional prior distribution
				Mat log_prior(y_max - y_min, x_max - x_min, CV_32F);
				Mat xv(1, x_max - x_min, CV_32F);   // The coordinates for which
				Mat yv(y_max - y_min, 1, CV_32F);   //  to evaluate the mvnpdf
				int ii = 0; for (int x = x_min; x<x_max; x++) { xv.at<float>(0, ii++) = x; }
				ii = 0;     for (int y = y_min; y<y_max; y++) { yv.at<float>(ii++, 0) = y; }
				Mat xx = repeat(xv, yv.total(), 1);
				Mat yy = repeat(yv, 1, xv.total());
				Mat coords(2, xx.total(), CV_32F);
				Mat xCol(coords(Range(0, 1), Range::all()));
				(xx.reshape(0, 1)).copyTo(xCol);
				Mat yCol(coords(Range(1, 2), Range::all()));
				(yy.reshape(0, 1)).copyTo(yCol);

				if (theta < 0)  {  // Use only the conditional prior
					log_prior = log_mvnpdf(coords, repeat(mu_cond, 1, coords.cols), C_cond).reshape(0, xx.rows);

				}
				else {  // Combine the conditional and dynamical prior
					mu_dyn.at<float>(0, 0) = float(glintPoints_prev[ORDERS[j][k]].x);
					mu_dyn.at<float>(1, 0) = float(glintPoints_prev[ORDERS[j][k]].y);
					Mat C_sum = C_cond + C_dyn;
					Mat log_scaling_factor = log_mvnpdf(mu_cond, mu_dyn, C_sum);  // This is actually a scalar
					Mat invC_cond(2, 2, CV_32F);
					if (invert(C_cond, invC_cond, DECOMP_CHOLESKY) == 0) { printf("C_conditional could not be inverted \n");  exit(-1); }
					Mat C_comb(2, 2, CV_32F);
					if (invert(invC_cond + invC_dyn, C_comb, DECOMP_CHOLESKY) == 0) { printf("C_combined could not be inverted \n");  exit(-1); }
					log_prior = log_scaling_factor.at<float>(0, 0) + log_mvnpdf(coords, repeat(C_comb * (invC_cond*mu_cond + invC_dyn*mu_dyn), 1, coords.cols), C_comb).reshape(0, xx.rows);
				}

				// Compute the logarithm of posterior distribution
				Mat log_post(y_max - y_min, x_max - x_min, CV_32F);
				log_post = log_prior + log_lhood;

				// The glint location shall be the MAP estimate
				minMaxLoc(log_post, &min_val, &max_val, &min_loc, &max_loc);
				glints_x[p][ORDERS[j][k]] = max_loc.x + x_min;
				glints_y[p][ORDERS[j][k]] = max_loc.y + y_min;
				log_post_MAP = log_post_MAP + max_val;

				//KL delete dynamically reserved arrays
				delete[] glints_x_sofar;
				delete[] glints_y_sofar;
				delete[] MU_X_sofar;
				delete[] MU_Y_sofar;

			}  // end of k loop

			if (log_post_MAP > largest_particle_weight) {
				best_particle = p;
				largest_particle_weight = log_post_MAP;
			}
			p++;  // increase the index of particle

		}  // end of j loop
	}  // end of n loop



	// Return the particle with the largest weight
	Point2d glintPoint;
	//std::vector<cv::Point2d> glintPoints;
	for (int i = 0; i<N_leds; i++) {
		glintPoint.x = glints_x[best_particle][i];
		glintPoint.y = glints_y[best_particle][i];
		glintPoints.push_back(glintPoint);
	}

}
