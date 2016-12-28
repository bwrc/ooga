#include "FrameProcessor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp> //eigen conversion
#include <iostream>

#include <thread>
#include "SG_common.h"

#include "kalmanFilterGazePoint.h"

#define DEFAULT_GAZE_DISTANCE 1.2

//FrameProcessor::FrameProcessor(concurrent_queue<std::shared_ptr<TBinocularFrame>>* in,
//	concurrent_queue<std::shared_ptr<TBinocularFrame>>* out)
FrameProcessor::FrameProcessor(BalancingQueue<std::shared_ptr<TBinocularFrame>>* in,
	BalancingQueue<std::shared_ptr<TBinocularFrame>>* out,
	TSettings* settings)
{
	this->qIn = in;
	this->qOut = out;

	etLeft = new EyeTracker();
	etRight = new EyeTracker();

	cv::FileStorage fs_params(settings->params_file, cv::FileStorage::READ);

	cv::Rect cropLeft;
	cv::Rect cropRight;

	fs_params["crop_left_x"] >> cropLeft.x;
	fs_params["crop_left_y"] >> cropLeft.y;
	fs_params["crop_left_w"] >> cropLeft.width;
	fs_params["crop_left_h"] >> cropLeft.height;

	fs_params["crop_right_x"] >> cropRight.x;
	fs_params["crop_right_y"] >> cropRight.y;
	fs_params["crop_right_w"] >> cropRight.width;
	fs_params["crop_right_h"] >> cropRight.height;

	etLeft->InitAndConfigure(FrameSrc::EYE_L, settings->CM_left_file, settings->glintmodel_file, settings->K9_file, settings->cam_left_eye_file, settings->LED_pos_file, cropLeft);
	etRight->InitAndConfigure(FrameSrc::EYE_R, settings->CM_right_file, settings->glintmodel_file, settings->K9_file, settings->cam_right_eye_file, settings->LED_pos_file, cropRight);

	fs_params["kalman_R_max"] >> kalman_R_max;

	//cv::FileStorage fs(K9_matrix_fn, cv::FileStorage::READ);
	cv::FileStorage fs(settings->K9_file, cv::FileStorage::READ);
	fs["K9_left"] >> K9_left;
	fs["K9_right"] >> K9_right;

	K9_left.convertTo(K9_left, CV_64F);
	K9_right.convertTo(K9_right, CV_64F);

	cv::FileStorage fs_cr(settings->camerarig_file, cv::FileStorage::READ);
	cv::Mat A_r2s_temp, A_l2r_temp;

	fs_cr["right_to_scene"] >> A_r2s_temp;
	fs_cr["left_to_right"] >> A_l2r_temp;

	A_r2s_temp.convertTo(A_r2s_temp, CV_64F);
	A_l2r_temp.convertTo(A_l2r_temp, CV_64F);

	cv::cv2eigen(A_r2s_temp, A_r2s);
	cv::cv2eigen(A_l2r_temp, A_l2r);


	// Calibration related stuff
	A_rot = (cv::Mat_<double>(3,3) << A_r2s(0,0), A_r2s(0,1), A_r2s(0,2), A_r2s(1,0), A_r2s(1,1), A_r2s(1,2), A_r2s(2,0), A_r2s(2,1), A_r2s(2,2) );
	a_tr = (cv::Mat_<double>(3,1) << A_r2s(0,3), A_r2s(1,3), A_r2s(2,3) );
	cv::invert(A_rot, invA_rot, cv::DECOMP_SVD);




	pauseWorking = false;

	//ptimer = new TPerformanceTimer();

	sceneCam = new Camera();

	cv::Mat scene_intrinsic;
	cv::Mat scene_dist;

	fs_cr["scene_intr"] >> scene_intrinsic;
	fs_cr["scene_dist"] >> scene_dist;

	scene_intrinsic.convertTo(scene_intrinsic, CV_64F);
	scene_dist.convertTo(scene_dist, CV_64F);

	sceneCam->setIntrinsicMatrix(scene_intrinsic);
	sceneCam->setDistortion(scene_dist);

	//param_est = cv::Mat_<double>(4, 1) << pog_scam.x, pog_scam.y, 0, 0;
	param_est = cv::Mat::zeros(4,1, CV_64F);
	P_est = cv::Mat::eye(4, 4, CV_64F);
	pog_scam_prev = cv::Point2d(320,240); // this shall be the initial value for Kalman filter
}

FrameProcessor::~FrameProcessor()
{
	if (etLeft != nullptr){
		delete etLeft;
	}
	if (etLeft != nullptr) {
		delete etRight;
	}

	delete sceneCam;

	//delete ptimer;
}

void FrameProcessor::blockWhilePaused(){
	std::unique_lock<std::mutex> lock(pauseMutex);
	while (pauseWorking)
	{
		pauseChanged.wait(lock);
	}
}

void FrameProcessor::start()
{
	running = true;
	zerotime = std::chrono::steady_clock::now();
	m_thread = std::thread(&FrameProcessor::Process, this);
}

void FrameProcessor::pause()
{
	std::unique_lock<std::mutex> lock(pauseMutex);
	pauseWorking = !pauseWorking; // this now functions as a toggle, rather than 'pause'

	pauseChanged.notify_all();
}

void FrameProcessor::stop()
{
	running = false;

	if (m_thread.joinable()){
		m_thread.join();
	}
}

void FrameProcessor::Process()
{
  hrclock::time_point _start = hrclock::now();

  while (running)
    {
      try{
	blockWhilePaused();

	//_start = hrclock::now();  // if commented, compute cumulative time; else, compute time per frame

	//TBinocularFrame* frame;
	auto frame = std::shared_ptr<TBinocularFrame>(nullptr);

	//if (qIn->try_dequeue(frame)){
	if (qIn->try_pop(frame)){

	  my_mtx.lock();

	  _start = hrclock::now();  // if commented, compute cumulative time; else, compute time per frame
	  hrclock::time_point processingTimeStart = hrclock::now();

	  cv::Point3d pupilCenter3DL, corneaCenter3DL;
	  cv::Point3d pupilCenter3DR, corneaCenter3DR;
	  double thetaL, thetaR;

	  /// Miikan muutos:
	  TTrackingResult* resL = new TTrackingResult;
	  TTrackingResult* resR = new TTrackingResult;

	  //runs left in it's own thread and right here
	  std::thread thr_eL = std::thread(&EyeTracker::Process, etLeft, frame->getImg(FrameSrc::EYE_L), std::ref(resL), std::ref(pupilCenter3DL), std::ref(corneaCenter3DL), std::ref(thetaL));
	  etRight->Process(frame->getImg(FrameSrc::EYE_R), resR, pupilCenter3DR, corneaCenter3DR, thetaR);
	  
	  thr_eL.join();

	  ///// Now we have results for both eyes --> combine
	  theta_mean = (thetaL + thetaR) / 2.0;  // self-explanatory

	  //transform 3D features from left to right camera coordinates for combining
	  Eigen::VectorXd cc_l2r(4), Cc3d_aug(4);
	  Cc3d_aug << resL->corneaCenter3D.x, resL->corneaCenter3D.y, resL->corneaCenter3D.z, 1;
	  cc_l2r = A_l2r * Cc3d_aug;
	  //TODO: retain the originals (CC3d in left camera coordinates)?
	  resL->corneaCenter3D.x = cc_l2r(0);
	  resL->corneaCenter3D.y = cc_l2r(1);
	  resL->corneaCenter3D.z = cc_l2r(2);

	  Eigen::VectorXd pc_l2r(4), Pc3d_aug(4);
	  Pc3d_aug << resL->pupilCenter3D.x, resL->pupilCenter3D.y, resL->pupilCenter3D.z, 1;
	  pc_l2r = A_l2r * Pc3d_aug;
	  resL->pupilCenter3D.x = pc_l2r(0);
	  resL->pupilCenter3D.y = pc_l2r(1);
	  resL->pupilCenter3D.z = pc_l2r(2);

	  cv::Point3d gazeVectorR;
	  cv::Point3d gazeVectorL;
	  cv::Point3d opticalVectorR;
	  cv::Point3d opticalVectorL;

	  // Multiply both eye's optical vectors by K matrix to get the gaze vector
	  opticalVectorL = (resL->pupilCenter3D - resL->corneaCenter3D) / double(cv::norm(resL->pupilCenter3D - resL->corneaCenter3D));
	  cv::Mat opticalVectorL_mat(opticalVectorL);
	  cv::Mat K9_times_optical_vector_mat_L = K9_left * opticalVectorL_mat;
	  cv::Point3d K9_times_optical_vector_point_L(K9_times_optical_vector_mat_L);
	  gazeVectorL = K9_times_optical_vector_point_L / cv::norm(K9_times_optical_vector_point_L);

	  opticalVectorR = (resR->pupilCenter3D - resR->corneaCenter3D) / double(cv::norm(resR->pupilCenter3D - resR->corneaCenter3D));
	  cv::Mat opticalVectorR_mat(opticalVectorR);
	  cv::Mat K9_times_optical_vector_mat_R = K9_right * opticalVectorR_mat;
	  cv::Point3d K9_times_optical_vector_point_R(K9_times_optical_vector_mat_R);//.at<float>(0), K9_times_optical_vector_mat_R.at<float>(1), K9_times_optical_vector_mat_R.at<float>(2));
	  gazeVectorR = K9_times_optical_vector_point_R / cv::norm(K9_times_optical_vector_point_R);

	  bool blinking = true;  // don't change this!
	  if (resR->score + resL->score > 0.7) {  // good scores --> not blinking
	    blinking = 0;
	  }

	  double gaze_dist = DEFAULT_GAZE_DISTANCE; //TODO: this should probably be instantiated elsewhere for control

	  // Check if features of both eyes are ok; otherwise, use only "better" eye or don't use either
	  bool using_both_eyes = 1;
	  cv::Point3d gazePoint_ecam;
	  int best_eye_ind = 0; //0 = right, 1 = left
	  double best_score = resL->score;
	  if (abs(resR->score - resL->score) > 0.2) {
	    using_both_eyes = 0;
	    if (resR->score >= resL->score) {
	      best_eye_ind = 0; best_score = resR->score;
	      gazePoint_ecam = resR->corneaCenter3D + gaze_dist * gazeVectorR;
	    }
	    else{
	      best_eye_ind = 1; best_score = resL->score;
	      gazePoint_ecam = resL->corneaCenter3D + gaze_dist * gazeVectorL;
	    }
	  }

	  //Note: when the gaze target is to the far left/right, the pupil sizes should differ (but not much...)!
	  if (std::abs((resR->pupilEllipse.size.height - resL->pupilEllipse.size.height)) > imgSize.y*0.1 ||
	      std::abs((resR->pupilEllipse.size.width - resL->pupilEllipse.size.width)) > imgSize.x*0.1) 
	    {
	      if (resR->pupilEllipse.size.height > resL->pupilEllipse.size.height) // assume that the smaller pupil is ok --> right eye
		{
		  if (using_both_eyes) {
		    using_both_eyes = 0;
		    best_eye_ind = 0;
		  }
		  else {
		    if (best_eye_ind = 1) {
		      blinking = 1; // seems that both eyes are "bad" (other miss glints and the other miss pupil) --> set blinking=1
		    }
		  }
		}
	      else { // left eye is ok
		if (using_both_eyes) {
		  using_both_eyes = 0;
		  best_eye_ind = 1;
		}
		else {
		  if (best_eye_ind = 0) {
		    blinking = 1; // seems that both eyes are "bad" (other miss glints and the other miss pupil) --> set blinking=1
		  }
		}
	      }
	    }
	  
	  if (resR->corneaCenter3D.z < 0) {  // sometimes the CC3D is on the other side of the eye (but this is veeeery rare, in practice only when blinking)
	    if (using_both_eyes) {
	      using_both_eyes = 0;
	      best_eye_ind = 1;
	    }
	    if (!using_both_eyes && best_eye_ind == 0) {  // both eyes are bad
	      blinking = 1;
	    }
	  }
	  if (resL->corneaCenter3D.z < 0) {
	    if (using_both_eyes) {
	      using_both_eyes = 0;
	      best_eye_ind = 0;
	    }
	    if (!using_both_eyes && best_eye_ind == 1) {  // both eyes are bad
	      blinking = 1;
	    }
	  }

	  if (using_both_eyes) {

	    cv::Point3d cr = resR->corneaCenter3D;  // Right cornea center
	    cv::Point3d cl = resL->corneaCenter3D;  // Left cornea center
	    cv::Point3d gr = gazeVectorR;           // Right gaze vector
	    cv::Point3d gl = gazeVectorL;           // Left gaze vector

	    // compute the gaze point as the 3d point with smallest square distance to the two gaze vectors. Results in noisy estimates, especially when looking further away.
	    // double gaze_dist_left = 1 / (1 - gr.dot(gl) * gl.dot(gr)) * (cr.dot(gl) - cl.dot(gl) + gr.dot(gl)*cl.dot(gr) - gr.dot(gl) * cr.dot(gr));  // "old" method, noisy...
	    // double gaze_dist_right = cl.dot(gr) - cr.dot(gr) + gaze_dist_left2*gl.dot(gr);

	    // Here the left and right gaze vectors are taken to be equally long which length is computed from simple trigonometry. This approach results in more robust estimate.
	    double gaze_dist_left = 0.5*norm(cr - cl) / sin(0.5*acos(gl.dot(gr)));
	    double gaze_dist_right = gaze_dist_left;

	    gazePoint_ecam = (cv::Point3d(cl + gaze_dist_left*gl) + cv::Point3d(cr + gaze_dist_right*gr)) / 2;  // The computed gaze point is in the right (user's right) eye cam coordinates
	  }

	  double pog_x, pog_y;

	  /// Map the gaze point to scene camera coordinates
	  Eigen::Vector4d gazePoint_ecam_aug(gazePoint_ecam.x, gazePoint_ecam.y, gazePoint_ecam.z, 1);
	  Eigen::Vector4d gazePoint_scam_aug = A_r2s * gazePoint_ecam_aug;
	  cv::Point3d gazePoint_scam = cv::Point3d(gazePoint_scam_aug(0), gazePoint_scam_aug(1), gazePoint_scam_aug(2));

	  cv::Point3d gazePoint_scam_norm = gazePoint_scam; // * 1 /norm(gazePoint_scam); (no need to normalize; gives the same result also without normalizing)
	  sceneCam->worldToPix(gazePoint_scam_norm, &pog_x, &pog_y);

	  //scene cam is upside down in the current frame
	  pog_x = imgSize.x - pog_x;
	  pog_y = imgSize.y - pog_y;

	  cv::Point2d pog_scam(pog_x, pog_y);

	  bool USE_KALMAN = 1;  // TODO: define elsewhere

	  cv::UMat* sceneImage = frame->getImg(FrameSrc::SCENE);

	  // TODO: do plotting elsewhere
	  if (USE_KALMAN) {
	    cv::circle(*sceneImage, pog_scam, 15, cv::Scalar(0, 150, 0), -1, 8);  // Plot the un-filtered point
	    cv::Point2d velo_meas = (pog_scam - pog_scam_prev) * theta_mean;  //  (theta_mean is averaged over L and R)
	    pog_scam_prev = pog_scam;
	    loc_variance = kalman_R_max * (1-theta_mean); // location observation variance
	    kalmanFilterGazePoint(pog_scam, velo_meas, &param_est, &P_est, loc_variance);  // theta <---> R_sigma ?
	    pog_scam.x = param_est.at<double>(0);
	    pog_scam.y = param_est.at<double>(1);
	  }


	  //// ----- The actual algorithms end (apart from some calibration related computations...) -----

	  bool USE_FOVEATED_SMOOTHING = false;
	  if (USE_FOVEATED_SMOOTHING) {
	    (*sceneImage).convertTo(*sceneImage, CV_32FC3);
	    cv::Mat sceneImage_blur;
	    cv::GaussianBlur(*sceneImage, sceneImage_blur, cv::Size(21, 21), 51);
	    float sigma2 = 10000;
	    int Svert = (*sceneImage).rows;
	    int Shori = (*sceneImage).cols;
	    cv::Mat tmp = (*sceneImage).getMat(cv::ACCESS_READ);
	    for (int i = 0; i<Svert; i++) {
	      for (int j = 0; j<Shori; j++) {
		for (int ch = 0; ch<3; ch++) {
		  float weight = exp(-((i - pog_scam.y)*(i - pog_scam.y) + (j - pog_scam.x)*(j - pog_scam.x)) / (2.0*sigma2));
		  tmp.at<cv::Vec3f>(i, j)[ch] = float(weight * float(tmp.at<cv::Vec3f>(i, j)[ch]) + \
		      (1 - weight) * (1 / 3.0 * (float(sceneImage_blur.at<cv::Vec3f>(i, j)[0]) + float(sceneImage_blur.at<cv::Vec3f>(i, j)[1]) + float(sceneImage_blur.at<cv::Vec3f>(i, j)[2]))));
		}
	      }
	    }
	    tmp.convertTo(*sceneImage, CV_8UC3);
	  }

	  // Draw the pog in the scene image. Adjust its size according to the glint fit score (of the "best camera"!).
	  cv::circle(*sceneImage, pog_scam, 35 * (1 - best_score) + 5, cv::Scalar(0, 0, 250), 7, 8);
	  
	  if (blinking) {
	    cv::putText(*sceneImage, "BLINK ", cv::Point2d(100, 300), CV_FONT_HERSHEY_PLAIN, 10, CV_RGB(250, 0, 100), 5);
	  }
	  if (!using_both_eyes) { // print text, if only other eye is used?
	    if (best_eye_ind == 0) {
	      cv::putText(*sceneImage, "(Using only right eye) ", cv::Point2d(200, 30), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(150, 150, 0), 2);
	    }
	    if (best_eye_ind == 1) {
	      cv::putText(*sceneImage, "(Using only left eye) ", cv::Point2d(50, 30), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(150, 150, 0), 2);
	    }
	  }

	  // TODO: plot elsewhere

	  cv::UMat* temp = frame->getImg(FrameSrc::EYE_R);

	  if (temp->size().width > 200) {
	    cv::circle(*temp, resR->pupilCenter2D,
		       6, cv::Scalar(0, 0, 255), -1, 8);

	    for (auto& g : resR->glintPoints) {
	      cv::circle(*temp, cv::Point2d(g.x, g.y),
			 6, cv::Scalar(255, 0, 0), -1, 8);
	    }
	    for (auto& p : resR->pupilEllipsePoints){
	    	cv::circle(*temp, cv::Point2d(p.x, p.y),
	    		6, cv::Scalar(0, 255, 0), -1, 8);
	    }
	  }
	  temp = frame->getImg(FrameSrc::EYE_L);

	  if (temp->size().width > 200) {
	    cv::circle(*temp, resL->pupilCenter2D,
		       6, cv::Scalar(0, 0, 255), -1, 8);

	    for (auto& g : resL->glintPoints) {
	      cv::circle(*temp, cv::Point2d(g.x, g.y),
			 6, cv::Scalar(255, 0, 0), -1, 8);
	    }
	    for (auto& p : resL->pupilEllipsePoints){
	    	cv::circle(*temp, cv::Point2d(p.x, p.y),
	    		6, cv::Scalar(0, 255, 0), -1, 8);
	    }
	  }

	  //qOut->enqueue(frame);
	  qOut->push(frame); //this doesn't probably need the balancing behavior? so could be a concurrent_queue as well..?

	  msecs processingTime = std::chrono::duration_cast<msecs>(hrclock::now() - processingTimeStart);

	  qIn->reportConsumerTime(processingTime.count());
	  my_mtx.unlock();

	  msecs frameTime = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
	  //std::cout << frameTime.count() << std::endl;  // print the computation time
	

	  // Save these into the  class variables, to be (possibly) used in the user calibration
	  corneaCenter3D_left = resL->corneaCenter3D;
	  pupilCenter3D_left = resL->pupilCenter3D;
	  corneaCenter3D_right = resR->corneaCenter3D;
	  pupilCenter3D_right = resR->pupilCenter3D;
	  if (!blinking && using_both_eyes) {
	    calibration_sample_ok = true; }
	  else {
	    calibration_sample_ok = false; }

	}

	msecs frameTime2 = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
	msecs waitTime = msecs(30) - frameTime2;

	//std::cout << "frametime: " << frameTime2.count() << std::endl;

	if (waitTime > msecs(0)){
	  std::this_thread::sleep_for(waitTime);
	}



      }
      catch (int e) {
	std::cout << "error in processor: " << e << std::endl;
      }

    }

}

void FrameProcessor::calibrationCallback( double x, double y)
{

  if (!calibration_sample_ok) { // Not a good sample (not using both eyes or blinking) ---> don't use
    std::cout << "Bad sample! Not to be used for calibration..." << std::endl;
    return; }


  if (N_cal_points == 0) {   // set these as unity matrices in the beginning (but this could be done elsewhere...):
    K9_left = cv::Mat::eye(3,3, K9_left.type());
    K9_right = cv::Mat::eye(3,3, K9_right.type());
  }

  N_cal_points++;

  int Shori = 640;  // TODO: define elsewhere!
  int Svert = 480;  // TODO: define elsewhere!
  double gaze_dist = 1.0;  // TODO: define elsewhere!
  cv::Point3d target3D;
  cv::Mat tmp;

  //std::cout << "fp would add " << x << ", " << y << std::endl;

  double mouse_X = double(Shori) - x;  // Our scene camera is upside down
  double mouse_Y = double(Svert) - y;

  sceneCam->pixToWorld(mouse_X, mouse_Y, target3D);
  target3D = target3D * gaze_dist;

  cv::Mat target3D_mat(target3D);
  tmp = target3D_mat.t();
  target3D_calpoints.push_back(tmp);


  cv::Mat corneaCenter3D_left_mat(corneaCenter3D_left);
  tmp = corneaCenter3D_left_mat.t();
  corneaCenter3D_left_calpoints.push_back(tmp);

  cv::Mat pupilCenter3D_left_mat(pupilCenter3D_left);
  cv::Mat opticalVector_left_mat = (pupilCenter3D_left_mat - corneaCenter3D_left_mat) / cv::norm((pupilCenter3D_left_mat - corneaCenter3D_left_mat));
  tmp = opticalVector_left_mat.t();
  opticalVector_left_calpoints.push_back(tmp);


  cv::Mat corneaCenter3D_right_mat(corneaCenter3D_right);
  tmp = corneaCenter3D_right_mat.t();
  corneaCenter3D_right_calpoints.push_back(tmp);

  cv::Mat pupilCenter3D_right_mat(pupilCenter3D_right);
  cv::Mat opticalVector_right_mat = (pupilCenter3D_right_mat - corneaCenter3D_right_mat) / cv::norm((pupilCenter3D_right_mat - corneaCenter3D_right_mat));
  tmp = opticalVector_right_mat.t();
  opticalVector_right_calpoints.push_back(tmp);


  if (N_cal_points >= 3) {  // When we have at least 3 samples, we can calibrate (as the matrices invert)
    cv::Mat a_tr_rep = cv::repeat(a_tr, 1, N_cal_points);
    cv::Mat invL_left, invL_right;  // inverses of the 'L' vectors (i.e., of the optical vectors)

    cv::invert(opticalVector_left_calpoints.t(), invL_left, cv::DECOMP_SVD);
    K9_left = 1.0 / gaze_dist * (invA_rot * (target3D_calpoints.t() - a_tr_rep) - corneaCenter3D_left_calpoints.t()) * invL_left;

    cv::invert(opticalVector_right_calpoints.t(), invL_right, cv::DECOMP_SVD);
    K9_right = 1.0 / gaze_dist * (invA_rot * (target3D_calpoints.t() - a_tr_rep) - corneaCenter3D_right_calpoints.t()) * invL_right;
  }

  std::cout << N_cal_points << " calibration samples collected ... " << std::endl;

}

