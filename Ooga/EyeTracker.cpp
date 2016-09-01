#include "EyeTracker.h"

#include "PerformanceTimer.h"

#include "getPupilCenter.h"

EyeTracker::EyeTracker()
{
	framecounter = 0;
	//	this->src = FrameSrc(0); //default, there's bound to be at least one cam
	pt = new TPerformanceTimer();
}


EyeTracker::~EyeTracker()
{
	delete pt;

	if (glintfinder != nullptr) {
		delete glintfinder;
	}
}

void EyeTracker::InitAndConfigure(FrameSrc myEye, std::string CM_fn, std::string glintmodel_fn)
{
	//TODO: read these from setting / function params
	int cols = 640;
	int rows = 480;
	this->setCropWindowSize(150, 100, 350, 300);
	lambda_ed = -0.02;  // initial guess (was -0.03)
	alpha_ed = 500;  // initial guess (was 300 or 200)
	theta = -1;
	theta_prev = -1;
	weight = 0.7;
	
	//allocate memory for mats
	gray = cv::UMat(rows, cols, CV_8UC1, cv::Scalar::all(0));
	cropped = cv::UMat(cropsizeX, cropsizeY, CV_8UC1, cv::Scalar::all(0));
	opened = cv::UMat(cropsizeX, cropsizeY, CV_8UC1, cv::Scalar::all(0));
	previous = cv::UMat(cropsizeY, cropsizeX, CV_8UC1, cv::Scalar::all(0));

	sigmoid_buffer = cv::Mat::zeros(1, 5, CV_64F);

	//create glint element
	int morph_radius = 3;
	glint_element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_radius + 1, 2 * morph_radius + 1));
	glint_element.at<uchar>(0, 0) = 0; glint_element.at<uchar>(0, 1) = 0; glint_element.at<uchar>(1, 0) = 0;
	glint_element.at<uchar>(glint_element.rows - 2, 0) = 0; glint_element.at<uchar>(glint_element.rows - 1, 0) = 0; glint_element.at<uchar>(glint_element.rows - 1, 1) = 0;
	glint_element.at<uchar>(glint_element.rows - 2, glint_element.cols - 1) = 0; glint_element.at<uchar>(glint_element.rows - 1, glint_element.cols - 2) = 0; glint_element.at<uchar>(glint_element.rows - 1, glint_element.cols - 1) = 0;
	glint_element.at<uchar>(0, glint_element.cols - 2) = 0; glint_element.at<uchar>(0, glint_element.cols - 1) = 0; glint_element.at<uchar>(1, glint_element.cols - 1) = 0;

	// Create a custom distance-based kernel for filtering pupil AND a SE for pupil
	int pupil_kernel_radius = 15; // Approximative pupil kernel (was 15)
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

	///// LOAD THE GLINT MODEL (MU and CM) ////// TODO: make this into a function
	// Mean of the glint model
	float MU_X[6], MU_Y[6], MU_X2[6], MU_Y2[6];
	cv::Mat MU_X_mat, MU_Y_mat, MU_X2_mat, MU_Y2_mat;

	// Load parameter values from parameters.yaml
	cv::FileStorage fs_gm(glintmodel_fn, cv::FileStorage::READ);
		if (myEye == FrameSrc::EYE_R) {
			fs_gm["mu_x_right"] >> MU_X_mat;
			fs_gm["mu_y_right"] >> MU_Y_mat;
		}
		else if(myEye == FrameSrc::EYE_L){
			fs_gm["mu_x_left"] >> MU_X_mat;
			fs_gm["mu_y_left"] >> MU_Y_mat;
		}

	int N_leds = 6;

	for (int i = 0; i < N_leds; i++) {
		MU_X[i] = MU_X_mat.at<float>(0, i);
		MU_Y[i] = MU_Y_mat.at<float>(0, i);
	}

	float scale = 1.0;  // NOTE!!
	for (int i = 0; i<N_leds; i++) {
		MU_X[i] = MU_X[i] * scale; 
		MU_Y[i] = MU_Y[i] * scale; 
	}

	// LOAD CM
	cv::Mat CM(N_leds * 2, N_leds * 2, CV_32F);

	//// Read the covariance matrises (CM) from files
	FILE *file_CM = fopen(CM_fn.c_str(), "r");
	if (file_CM == 0) {
		printf("Covariance file not found \n");
		exit(-1);
	}
	fseek(file_CM, 0, SEEK_END); 
	long lSize = ftell(file_CM); rewind(file_CM);
	char *buffer_CM = (char*)malloc(sizeof(char)*lSize);   // allocate memory to contain the whole file
	if (lSize<N_leds*N_leds * 4) { printf("Not a full N_leds² x N_leds² covariance matrix \n"); exit(-1); }
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

	glintfinder = new GlintFinder();
	glintfinder->Initialize( CM, MU_X, MU_Y, 6 );

	//reset prev for first round
	for (int i = 0; i < 6; ++i) {
		glintPoints_prev.push_back(cv::Point2d(0, 0));
	}


}

void EyeTracker::setCropWindowSize(int xmin, int ymin, int width, int height)
{
	this->cropminX = xmin;
	this->cropminY = ymin;
	this->cropsizeX = width;
	this->cropsizeY= height;
}

void EyeTracker::Process(cv::UMat* eyeframe, TTrackingResult &trackres, cv::Point3d &pupilCenter3D, cv::Point3d &corneaCenter3D)
{
	pt->start();

	//TODO: just do this for the cropped part?
	cv::cvtColor((*eyeframe), gray, cv::COLOR_BGR2GRAY);

	std::cerr << eyeframe->getMat(cv::ACCESS_READ).size() << std::endl;
	std::cerr << "pix1: " << int(eyeframe->getMat(cv::ACCESS_READ).at<uchar>(199, 123)) << std::endl;
	std::cerr << "pix2: " << int(eyeframe->getMat(cv::ACCESS_READ).at<uchar>(198, 124)) << std::endl;


	pt->addTimeStamp("converted");

	//if ganzheit frames (through mirror), flip eye image!
	//flip( gray, gray );

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

	// Get the eye image difference and 'sigmoid' it (only for the right eye! 
	//<- TODO: here it's for both as eyetrackers are independent)
	cv::subtract(filtered, previous, imageDiff);
	double eye_diff = norm(imageDiff);
	double sigmoid = 1.0 / (1 + exp(lambda_ed * (eye_diff - alpha_ed)));

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

	//moved these
	//float MU_X[6], MU_Y[6];
	////cv::Mat CM;
	////this is a debug attempt, remove:
	////TODOTODO: load CM (etc) from file 
	//cv::Mat CM(12,12, CV_32F);
	//CM.setTo(0.5);
	////end debug attempt

	////TODOTODO: move CM, MU_X, MU_Y (check left and right eye!) to a glintFinder Init function

	double score;
	float glint_scores[6];
	float glint_beta = 100.0f;  // The coefficient of the glint likelihood model
	float glint_reg_coef = 1.0f;   // The initial regularization coefficient for the covariance matrix (if zero, might result in non-valid covariance matrix ---> crash)

	std::vector<cv::Point2d> glintpoints;// (6, cv::Point2d(0, 0));
	//silly debug, remove (how does miika do this)?
	//for (int i = 0; i < 6; ++i) {
	//	glintPoints_prev.push_back(cv::Point2d(0, 0));
	//}

	//TODO: check MU-args for eyeness (the tracker should relate to one, init function should select)
	glintpoints = glintfinder->getGlints(diffimg, pupil_center, glintPoints_prev,
		theta, /*MU_X, MU_Y, CM, moved to glintfinder*/ glint_kernel, score, glint_scores, glint_beta, glint_reg_coef);
	glintPoints_prev = glintpoints;
	pt->addTimeStamp("glintpoints");

	std::cout << glintPoints << std::endl;

	//TODO: insert tracking results to frame, visualize in main thread
	if (eyeframe->size().width > 200) {
		circle(*eyeframe, cv::Point2d(pupil_center.x + cropminX, pupil_center.y + cropminY),
			6, cv::Scalar(0, 0, 255), -1, 8);
		for (auto& g : glintpoints) {
			circle(*eyeframe, cv::Point2d(g.x + cropminX, g.y + cropminY),
				6, cv::Scalar(255,0,0), -1, 8);

		}
	}
	pt->addTimeStamp("draw");

	//dumping stuff to cout takes 10ms?!
	//pt->dumpTimeStamps(std::cout);

}
