#include "EyeTracker.h"

#include "PerformanceTimer.h"

#include "getPupilCenter.h"

#include "computePupilCenter3d.h"

#include "SG_common.h"

EyeTracker::EyeTracker()
{
	framecounter = 0;
	pt = new TPerformanceTimer();
}


EyeTracker::~EyeTracker()
{
	delete pt;

	if (glintfinder != nullptr) {
		delete glintfinder;
	}
}

void EyeTracker::InitAndConfigure(FrameSrc myEye, 
                                  std::string CM_fn, 
																	std::string glintmodel_fn, 
																	std::string K9_matrix_fn, 
																	std::string cam_cal_fn,
																	std::string LED_pos_fn,
																	cv::Rect cropWindow)
{
	//TODO: read all these from setting / function params:
	int cols = 640;
	int rows = 480;
	//this->setCropWindowSize(150, 100, 350, 350);  // todo: we could crop heavier
	//this->setCropWindowSize(170, 110, 250, 210);  // a heavier crop!
	this->setCropWindowSize(cropWindow.x, cropWindow.y, cropWindow.width, cropWindow.height);
	lambda_ed = 0.02;  // initial guess
	alpha_ed = 500;  // initial guess
	weight = 0.7;

	theta_prev = -1;

	//allocate memory for mats
	gray = cv::UMat(rows, cols, CV_8UC1, cv::Scalar::all(0));
	cropped = cv::UMat(cropsizeX, cropsizeY, CV_8UC1, cv::Scalar::all(0));
	opened = cv::UMat(cropsizeX, cropsizeY, CV_8UC1, cv::Scalar::all(0));
	previous = cv::UMat(cropsizeY, cropsizeX, CV_8UC1, cv::Scalar::all(0));

	sigmoid_buffer = cv::Mat::zeros(1, 5, CV_64F);

	//create glint element
	int morph_radius = 5;
	glint_element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_radius + 1, 2 * morph_radius + 1));
	glint_element.at<uchar>(0, 0) = 0; glint_element.at<uchar>(0, 1) = 0; glint_element.at<uchar>(1, 0) = 0;
	glint_element.at<uchar>(glint_element.rows - 2, 0) = 0; glint_element.at<uchar>(glint_element.rows - 1, 0) = 0; glint_element.at<uchar>(glint_element.rows - 1, 1) = 0;
	glint_element.at<uchar>(glint_element.rows - 2, glint_element.cols - 1) = 0; glint_element.at<uchar>(glint_element.rows - 1, glint_element.cols - 2) = 0; glint_element.at<uchar>(glint_element.rows - 1, glint_element.cols - 1) = 0;
	glint_element.at<uchar>(0, glint_element.cols - 2) = 0; glint_element.at<uchar>(0, glint_element.cols - 1) = 0; glint_element.at<uchar>(1, glint_element.cols - 1) = 0;

	// Create a custom distance-based kernel for filtering pupil AND a SE for pupil
	int pupil_kernel_radius = 15; // Approximative pupil kernel
	pupil_kernel = cv::Mat::ones(2 * pupil_kernel_radius + 1, 2 * pupil_kernel_radius + 1, CV_32F);
	pupil_element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * pupil_kernel_radius + 1, 2 * pupil_kernel_radius + 1));
	for (int i = 0; i<2 * pupil_kernel_radius + 1; i++) {
		for (int j = 0; j<2 * pupil_kernel_radius + 1; j++) {
			if (sqrt((pupil_kernel_radius - i)*(pupil_kernel_radius - i) + (pupil_kernel_radius - j)*(pupil_kernel_radius - j)) >= pupil_kernel_radius) {
				pupil_kernel.at<float>(i, j) = float(0);
				pupil_element.at<uchar>(i, j) = 0;
			}
		}
	}
	cv::normalize(pupil_kernel, pupil_kernel, 1, 0, cv::NORM_L1);

	// Create another kernel for pupil edge locating
	int pupil_kernel_radius2 = 5;
	pupil_kernel2 = cv::Mat::ones(2 * pupil_kernel_radius2 + 1, 2 * pupil_kernel_radius2 + 1, CV_32F);
	for (int i = 0; i<2 * pupil_kernel_radius2 + 1; i++) {
		for (int j = 0; j<2 * pupil_kernel_radius2 + 1; j++) {
			if (sqrt((pupil_kernel_radius2 - i)*(pupil_kernel_radius2 - i) + (pupil_kernel_radius2 - j)*(pupil_kernel_radius2 - j)) >= pupil_kernel_radius2) {
				pupil_kernel2.at<float>(i, j) = float(0);
			}
		}
	}
	cv::normalize(pupil_kernel2, pupil_kernel2, 1, 0, cv::NORM_L1);

	// Create a custom distance-based kernel for filtering glints
	int glint_kernel_radius = 2; // Approximative glint kernel
	glint_kernel = cv::Mat::ones(2 * glint_kernel_radius + 1, 2 * glint_kernel_radius + 1, CV_32F);
	for (int i = 0; i<2 * glint_kernel_radius + 1; i++) {
		for (int j = 0; j<2 * glint_kernel_radius + 1; j++) {
			glint_kernel.at<float>(i, j) = (sqrt(2 * glint_kernel_radius*glint_kernel_radius) - sqrt((glint_kernel_radius - i)*(glint_kernel_radius - i) + (glint_kernel_radius - j)*(glint_kernel_radius - j)));
		}
	}
	cv::normalize(glint_kernel, glint_kernel, 1, 0, cv::NORM_L1);

	///// LOAD THE GLINT MODEL (MU and CM)
	// Mean of the glint model
	float MU_X[6], MU_Y[6], MU_X2[6], MU_Y2[6];
	cv::Mat MU_X_mat, MU_Y_mat, MU_X2_mat, MU_Y2_mat;
	cv::FileStorage fs_gm(glintmodel_fn, cv::FileStorage::READ);
	if (myEye == FrameSrc::EYE_R) {
	  fs_gm["mu_x_right"] >> MU_X_mat;
	  fs_gm["mu_y_right"] >> MU_Y_mat;
	}
	else if(myEye == FrameSrc::EYE_L){
	  fs_gm["mu_x_left"] >> MU_X_mat;
	  fs_gm["mu_y_left"] >> MU_Y_mat;
	}
	
	int N_leds = 6;  // TODO: define this elsewhere

	for (int i = 0; i < N_leds; i++) {
		MU_X[i] = MU_X_mat.at<float>(0, i);
		MU_Y[i] = MU_Y_mat.at<float>(0, i);
	}

	// Load parameter values from parameters.yaml
	cv::FileStorage fs("../calibration/parameters.yaml", cv::FileStorage::READ);
	fs["glint_beta"] >> glint_beta;
	fs["glint_reg_coef"] >> glint_reg_coef;
	fs["pupil_iterate"] >> pupil_iterate;
	fs["pupil_beta"] >> pupil_beta;

	// LOAD CM
	cv::Mat CM(N_leds * 2, N_leds * 2, CV_32F);

	//// Read the covariance matrices (CM) from files
	//TODO move this to a .yaml file
	FILE *file_CM = fopen(CM_fn.c_str(), "r");
	if (file_CM == 0) {
	  printf("Covariance file not found \n");
	  exit(-1);
	}
	fseek(file_CM, 0, SEEK_END);
	long lSize = ftell(file_CM); rewind(file_CM);
	char *buffer_CM = (char*)malloc(sizeof(char)*lSize);   // allocate memory to contain the whole file
	if (lSize<N_leds*N_leds * 4) { printf("Not a full 2xN_leds x 2xN_leds  covariance matrix \n"); exit(-1); }
	if (buffer_CM == NULL) { fputs("Memory error", stderr); exit(2); }
	fread(buffer_CM, 1, lSize, file_CM);   // read the number
	fclose(file_CM);
	// Parse the CM buffer:
	int i = 0;
	for (int r = 0; r<N_leds * 2; r++) {
	  for (int c = 0; c<N_leds * 2; c++) {
	    CM.at<float>(r, c) = atof(&buffer_CM[i]);
	    while (1) {
	      i++;
	      if (buffer_CM[i] == ',') {
		i++;
		break;
	      }
	    }
	  }
	}
	free(buffer_CM);

	///////// END LOADING

	eyeCam = new Camera();

	cv::FileStorage camcalfs(cam_cal_fn, cv::FileStorage::READ);
	cv::Mat eye_intrinsic;
	cv::Mat eye_dist;

	camcalfs["eye_intr"] >> eye_intrinsic;
	camcalfs["eye_dist"] >> eye_dist;

	eye_intrinsic.convertTo(eye_intrinsic, CV_64F);
	eye_dist.convertTo(eye_dist, CV_64F);

	eyeCam->setIntrinsicMatrix(eye_intrinsic);
	eyeCam->setDistortion(eye_dist);

	glintfinder = new GlintFinder();
	//TODO: remove hardcoded values like the six (= Nleds) here
	glintfinder->Initialize( CM, MU_X, MU_Y, 6 );

	for (int i = 0; i < 6; ++i) {
	  glintPoints_prev.push_back(cv::Point2d(0, 0));
	}

	pupilestimator = new PupilEstimator();
	for (int i = 0; i < 4; i++) { // there are 4 endpoints of ellipse's axes
	  pupilEllipsePoints_prev_eyecam[i] = cv::Point2d(0, 0);
	}

  //Read LED positions
	cv::FileStorage LEDpos_fs(LED_pos_fn, cv::FileStorage::READ);
	cv::Mat ledpmat;
	switch (myEye ){
		case FrameSrc::EYE_L:
		LEDpos_fs["left_leds"] >> ledpmat;
		break;
		default: //right
		LEDpos_fs["right_leds"] >> ledpmat;
		break;
	}
	for(int i=0; i < ledpmat.rows; ++i){
		Eigen::Vector3d tmp;
		tmp << ledpmat.at<float>(i,0), ledpmat.at<float>(i,1), ledpmat.at<float>(i,2);
		std::cout << tmp << std::endl;
		led_pos.push_back(tmp);
	}

}

void EyeTracker::setCropWindowSize(int xmin, int ymin, int width, int height)
{
  this->cropminX = std::max(0, std::min(xmin, 640));
  this->cropminY = std::max(0, std::min(ymin, 480));
  this->cropsizeX = std::max(1, std::min(width,  640-this->cropminX));
  this->cropsizeY = std::max(1, std::min(height, 480-this->cropminY));
}

void EyeTracker::Process(cv::UMat* eyeframe, TTrackingResult* trackres, cv::Point3d &pupilCenter3D, cv::Point3d &corneaCenter3D, double &theta)
{

  // TODO: Miksi tämä "palauttaa" &pupilCenter3D ja &corneaCenter3D, kun ne kuitenkin ovat trackres -tietueen kenttiä? ---> Laita tulokset trackingres:iin.

  pt->start();

  //TODO: just do this for the cropped part?
  cv::cvtColor((*eyeframe), gray, cv::COLOR_BGR2GRAY);

  pt->addTimeStamp("converted");

  // crop
  cropped = gray(cv::Rect(cropminX, cropminY, cropsizeX, cropsizeY)).clone();
  // Remove the glint candidates by opening the image
  morphologyEx(cropped, opened, cv::MORPH_OPEN, glint_element);
  // Form a difference image where the glint candidates clearly stand out ("top-hat operation")
  cv::subtract(cropped, opened, diffimg);
  // Filter with pupil kernel
  filter2D(opened, filtered, -1, pupil_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

  pt->addTimeStamp("initial filters run");

  //// Compute the fixation parameter 'theta' (theta=0 --> fixation / theta=1 --> saccade or blink)

  cv::subtract(filtered, previous, imageDiff);
  double eye_diff = norm(imageDiff);
  double sigmoid = 1.0 / (1 + exp(-lambda_ed * (eye_diff - alpha_ed)));

  // Compute the maximum value of the sigmoid buffer
  for (int i = 0; i<sigmoid_buffer.cols - 1; i++) {
    sigmoid_buffer.at<double>(i) = sigmoid_buffer.at<double>(i + 1);
  }
  sigmoid_buffer.at<double>(sigmoid_buffer.cols - 1) = sigmoid;
  double min_val, max_sigmoid_buffer;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(sigmoid_buffer, &min_val, &max_sigmoid_buffer, &min_loc, &max_loc);

  // Transform the max_sigmoid_buffer into 'theta'
  if (max_sigmoid_buffer < theta_prev) {
    theta = weight*max_sigmoid_buffer + (1 - weight)*theta_prev;
  }
  else {
    theta = max_sigmoid_buffer;
  }
  theta = std::max(theta, double(1e-10));
  theta = std::min(theta, double(1));
  theta_prev = theta;

  pt->addTimeStamp("sigmoided");
  previous = filtered.clone();

  cv::Point2d pupil_center = getPupilCenter(filtered.getMat(cv::ACCESS_READ)); //approx pupil center

  pt->addTimeStamp("pupil center");

  double score;
  float glint_scores[6];
  //TODO: move to config file
  float glint_beta = 100.0f;  // The coefficient of the glint likelihood model
  float glint_reg_coef = 1.0f;   // The initial regularization coefficient for the covariance matrix (if zero, might result in non-valid covariance matrix ---> crash)

  std::vector<cv::Point2d> glintPoints;// (6, cv::Point2d(0, 0));

  glintPoints = glintfinder->getGlints(diffimg,
				       pupil_center, glintPoints_prev, theta, glint_kernel,
				       score, glint_scores, glint_beta, glint_reg_coef,
				       2, true);  // TODO: N_glint_candidates as "true" argument (it doesn't do shit now)
	
  glintPoints_prev = glintPoints;
  pt->addTimeStamp("glintpoints");

  //calculate cornea
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > glint_pos;
  Eigen::Vector3d eigGlint;
  for (int i = 0; i < 6; i++) {
    glintPoints[i].x = glintPoints[i].x + cropminX; //x_min;
    glintPoints[i].y = glintPoints[i].y + cropminY; // y_min;
    cv::Point3d glintPoint3d;
    eyeCam->pixToWorld(glintPoints[i].x, glintPoints[i].y, glintPoint3d);
    eigGlint << glintPoint3d.x, glintPoint3d.y, glintPoint3d.z;
    glint_pos.push_back(eigGlint);
  }

  cv::RotatedRect pupilEllipse;// = new cv::RotatedRect;

  pupilEllipse = pupilestimator->getPupilEllipse(opened.getMat(cv::ACCESS_READ), pupil_center, pupil_kernel2, pupil_element, pupil_iterate, pupil_beta);

  //TODO move this to pupilestimator for cleansiness?
  cv::Point2d pupilEllipsePoints[4];  // There are four endpoints in an ellipse's axes
  getPupilEllipsePoints(pupilEllipse, pupilEllipsePoints_prev_eyecam, double(theta), &pupilEllipsePoints[0]);

	
  for (int i=0; i<4; i++)  {  // Loop the four endpoints of the ellipse's axes
    pupilEllipsePoints_prev_eyecam[i] = pupilEllipsePoints[i];
  }

  for (int i = 0; i < 4; i++) {
    pupilEllipsePoints[i].x = pupilEllipsePoints[i].x + cropminX;
    pupilEllipsePoints[i].y = pupilEllipsePoints[i].y + cropminY;
  }

  std::vector<cv::Point3d> pupilEllipsePoints3d;
  for (int i = 0; i < 4; i++) {
    cv::Point3d pupilEllipsePoint3d;
    eyeCam->pixToWorld(pupilEllipsePoints[i].x, pupilEllipsePoints[i].y, pupilEllipsePoint3d);
    pupilEllipsePoints3d.push_back(pupilEllipsePoint3d);
  }

  const double gx_guess = 0.01;  // Insert something slightly positive here as otherwise the local solution may be at wrong side of the camera!
  std::vector<double> guesses(6, gx_guess);
  double error;
  Eigen::Vector3d eigCenter;

  ooga::Cornea *cornea = new ooga::Cornea();

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > led_pos_sub;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > glint_pos_sub;
  std::vector<double> guesses_sub;
  float glint_score_threshold = 0.1; //0.3;

  for (int i = 0; i < 6; i++) {  // TODO: replace 6 ---> Nleds
    if (glint_scores[i] > glint_score_threshold) {
      glint_pos_sub.push_back(glint_pos[i]);
      led_pos_sub.push_back(led_pos[i]);
      guesses_sub.push_back(gx_guess);
    }
  }

  if (glint_pos_sub.size() < 2) {
    cornea->computeCentre(led_pos, glint_pos, guesses, eigCenter, error);
  }
  else {
    cornea->computeCentre(led_pos_sub, glint_pos_sub, guesses_sub, eigCenter, error);  // Cc3d computer could be score-weighted!
  }
  Cc3d.x = eigCenter(0); Cc3d.y = eigCenter(1); Cc3d.z = eigCenter(2);

  /// Compute the 3D pupil center
  Pc3d = computePupilCenter3d(pupilEllipsePoints3d, Cc3d);

  /// Compute the pupil-corneal distance vector ("gaze vector") and correct it with K9
  cv::Point3f optical_vector = (Pc3d - Cc3d) / float(cv::norm(Pc3d - Cc3d));
  cv::Mat optical_vector_mat(optical_vector);

  //copy results to tracking results
  trackres->pupilCenter2D = cv::Point2d(pupil_center.x + cropminX, pupil_center.y + cropminY);
  trackres->pupilCenter3D = Pc3d;
  trackres->pupilEllipsePoints.reserve(4);
  for (int i = 0; i < 4; i++) {
    trackres->pupilEllipsePoints.push_back(pupilEllipsePoints[i]);
  }

  trackres->pupilEllipse = cv::RotatedRect(cv::Point2f(pupilEllipse.center.x+cropminX + pupilEllipse.center.y+cropminY), pupilEllipse.size, pupilEllipse.angle);

  trackres->glintPoints = glintPoints;
  trackres->corneaCenter3D = Cc3d;
  trackres->score = score;

  //TODO: insert tracking results to frame, visualize in main thread

  pt->addTimeStamp("draw");
  //pt->dumpTimeStamps(std::cout);

}
