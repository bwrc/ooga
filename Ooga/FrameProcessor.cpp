#include "FrameProcessor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <thread>
#include "SG_common.h"

#include "kalmanFilterGazePoint.h"

#define DEFAULT_GAZE_DISTANCE 1.2

//FrameProcessor::FrameProcessor(concurrent_queue<std::shared_ptr<TBinocularFrame>>* in,
//	concurrent_queue<std::shared_ptr<TBinocularFrame>>* out)
FrameProcessor::FrameProcessor(BalancingQueue<std::shared_ptr<TBinocularFrame>>* in,
	BalancingQueue<std::shared_ptr<TBinocularFrame>>* out)
{
	this->qIn = in;
	this->qOut = out;

	etLeft = new EyeTracker();
	etRight = new EyeTracker();

	//TODO Move hard-coded files to settings
	std::string CM_fn_left = "../calibration/file_CM_left";
	std::string CM_fn_right = "../calibration/file_CM_right";
	std::string glintmodel_fn = "../calibration/glint_model.yaml";
	std::string K9_matrix_fn = "../calibration/K9.yaml";
	std::string parameters_fn = "../calibration/parameters.yaml";

	etLeft->InitAndConfigure( FrameSrc::EYE_L, CM_fn_left, glintmodel_fn, K9_matrix_fn );
	etRight->InitAndConfigure(FrameSrc::EYE_R, CM_fn_right, glintmodel_fn, K9_matrix_fn );

	cv::FileStorage fs(K9_matrix_fn, cv::FileStorage::READ);
	fs["K9_left"] >> K9_left;
	fs["K9_right"] >> K9_right;

	K9_left.convertTo(K9_left, CV_64F);
	K9_right.convertTo(K9_right, CV_64F);

	//st = new TSceneTracker();
	//st->SetCamFeedSource(FrameSrc::SCENE);
	//st->InitAndConfigure(); // Doesn't read settings now. t: Miika

	pauseWorking = false;

	//ptimer = new TPerformanceTimer();

	//TODO: Load these from config files

	//// These are the matrices that transform (augmented) coordinates from the eye camera to the scene camera
	// A_l2s << 0.999527, -0.010507, -0.028904, 0.024822,     (useless!)
	// 		0.002372, -0.910701, 0.413060, -0.050344,
	// 		-0.030663, -0.412933, -0.910245, 0.005679,
	// 		0.000000, 0.000000, 0.000000, 1.000000;

	// Estimated with the optics-based hardware calibration scheme:
	// A_r2s << 0.999970, -0.007446, 0.002042, -0.032355,
	// 		-0.007611, -0.906227, 0.422724, -0.042055,
	// 		-0.001297, -0.422726, -0.906256, -0.001406,
	// 		0.000000, 0.000000, 0.000000, 1.000000;

	// Computed from the 3D model (one rotation of 23 degrees + translation):
	A_r2s << 1.000000, -0.000000, -0.000000, -0.031000, 
	  0.000000, -0.920505, 0.390731, -0.029422, 
	  0.000000, -0.390731, -0.920505, 0.023737, 
	  0.000000, 0.000000, 0.000000, 1.000000;
	

	//// the transformation from left eye camera to the right eye camera

	// Optics based measures:
	// A_l2r << 0.999519, -0.003039, -0.030866, 0.057229,
	// 	0.003370, 0.999937, 0.010674, 0.004091,
	// 	0.030832, -0.010772, 0.999467, -0.009809,
	// 	0.000000, 0.000000, 0.000000, 1.000000;

	// 3D model based measures:
	A_l2r << 1, 0, 0, 0.062,
	  0, 1, 0, 0,
	  0, 0, 1, 0,
	  0, 0, 0, 1;


	//SETUP CAMERA MATRICES
	//TODO: read these from cal file
	// Silmäkamerat ovat tässä turhia (ja L ja R oli väärinpäin) t: Miika

	// eyeCamL = new Camera();
	// eyeCamR = new Camera();
	sceneCam = new Camera();

	//left
	// double eye_intrL[9] = { 706.016649148281, 0, 0, 0, 701.594050122496, 0, 325.516892970862, 229.074499749446, 1 };
	// double eye_distL[5] = { -0.0592552807088912, -0.356035882388608, -0.00499637342440711, -0.00186924287347176, 0.041261857952091 };
	// eyeCamL->setIntrinsicMatrix(eye_intrL);
	// eyeCamL->setDistortion(eye_distL);
	// //right
	// double eye_intrR[9] = { 789.311956305243, 0, 0, 0, 785.608590236097, 0, 318.745586281075, 217.585069948245, 1 };
	// double eye_distR[5] = { -0.0683374878811475, -0.57464673534425, 0.00189640729826507, 0.00224588599102401, 0.61174675760327 };
	// eyeCamR->setIntrinsicMatrix(eye_intrR);
	// eyeCamR->setDistortion(eye_distR);

	//scene
	double scene_intr[9] = { 611.3922, 0, 0, 0, 613.3425, 0, 321.8853, 239.7948, 1.0000 };
	double scene_dist[5] = { 0.026858, -0.212083, 0.002701, -0.002490, 0.494850 };
	sceneCam->setIntrinsicMatrix(scene_intr);
	sceneCam->setDistortion(scene_dist);


	// I made it read it here (Miika)
	cv::FileStorage fs2(parameters_fn, cv::FileStorage::READ);
	fs2["kalman_R_max"] >> kalman_R_max;

	//param_est = cv::Mat_<double>(4, 1) << pog_scam.x, pog_scam.y, 0, 0;
	param_est = cv::Mat::zeros(4,1, CV_64F);
	P_est = cv::Mat::eye(4, 4, CV_64F);
	pog_scam_prev = cv::Point2d(320,240); // TODO: This should be "pog_scam" the first time!



}

FrameProcessor::~FrameProcessor()
{
	if (etLeft != nullptr){
		delete etLeft;
	}
	if (etLeft != nullptr) {
		delete etRight;
	}

	delete eyeCamL;
	delete eyeCamR;
	delete sceneCam;

	//if (st != nullptr){
	//	delete st;
	//}

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
	  //				cv::Mat K9_matrix = et->K9_matrix;
	  //				Eigen::Matrix4d A = st->A_matrix;
	  //				Camera sceneCam = st->sceneCam;

	  TTrackingResult* resL = new TTrackingResult;
	  TTrackingResult* resR = new TTrackingResult;

	  //runs left in it's own thread and right here
	  //std::thread thr_eL = std::thread(&EyeTracker::Process, etLeft, frame->getImg(FrameSrc::EYE_L), resL, pupilCenter3DL, corneaCenter3DL);
	  std::thread thr_eL = std::thread(&EyeTracker::Process, etLeft, frame->getImg(FrameSrc::EYE_L), std::ref(resL), std::ref(pupilCenter3DL), std::ref(corneaCenter3DL), std::ref(thetaL));
	  //etLeft->Process(frame->getImg(FrameSrc::EYE_L), resL, pupilCenter3DL, corneaCenter3DL);

	  //std::thread thr_eR = std::thread(&EyeTracker::Process, etRight, frame->getImg(FrameSrc::EYE_R), resR, pupilCenter3DL, corneaCenter3DL);
	  etRight->Process(frame->getImg(FrameSrc::EYE_R), resR, pupilCenter3DR, corneaCenter3DR, thetaR);


	  thr_eL.join();
	  //thr_eR.join();

	  ///// Now we have results for both eyes --> combine

	  theta_mean = (thetaL + thetaR) / 2.0;  // self-explanatory

	  //transform 3D features from left to right camera coordinates for combining
	  Eigen::VectorXd cc_l2r(4), Cc3d_aug(4);
	  Cc3d_aug << resL->corneaCenter3D.x, resL->corneaCenter3D.y, resL->corneaCenter3D.z, 1;
	  cc_l2r = A_l2r * Cc3d_aug;
	  //TODO: retain the originals?
	  resL->corneaCenter3D.x = cc_l2r(0);
	  resL->corneaCenter3D.y = cc_l2r(1);
	  resL->corneaCenter3D.z = cc_l2r(2);

	  Eigen::VectorXd pc_l2r(4), Pc3d_aug(4); // (3d pupil centers are actually just for calibration)
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


	  // K9_times_optical_vector_L = A_l2r * K9_times_optical_vector_aug_L;
	  // gazeVectorL = K9_times_optical_vector_point_L / norm(K9_times_optical_vector_point_L);

	  bool USE_KAPPA_CORRECTION = false;  // DON'T USE "KAPPA_CORRECTION", IT WAS JUST A TEST!

	  /// TODO: DON'T USE "KAPPA_CORRECTION", IT WAS JUST A TEST
	  /// TODO: ADD USER CALIBRATION!
	  // if (USE_KAPPA_CORRECTION){
	  //   cv::Mat x_vec = cv::Mat(cv::Point3f(resR->corneaCenter3D - resL->corneaCenter3D ));
	  //   cv::Mat y_vec = cv::Mat(cv::Point3f(resR->pupilCenter3D - resL->pupilCenter3D)).cross(x_vec);
	  //   cv::Mat z_vec = x_vec.cross(y_vec);
	  //   x_vec = x_vec / norm(x_vec);
	  //   y_vec = y_vec / norm(y_vec);
	  //   z_vec = z_vec / norm(z_vec);
	  //   cv::Mat xy; cv::hconcat(x_vec, y_vec, xy);
	  //   cv::Mat R_aug2world; cv::hconcat(xy, z_vec, R_aug2world);
	  //   cv::Mat R_world2aug = R_aug2world.inv();

	  //   //right eye
	  //   cv::Point3f pupil_cornea_vectorR = cv::Point3f(resR->pupilCenter3D - resR->corneaCenter3D);
	  //   cv::Mat gaze_vec_in_augR = R_world2aug * cv::Mat(pupil_cornea_vectorR) / norm(pupil_cornea_vectorR);
	  //   cv::Mat gaze_vec_in_aug_corrR = K9_right * gaze_vec_in_augR;
	  //   cv::Mat gaze_vec_in_worldR = R_aug2world * gaze_vec_in_aug_corrR;

	  //   //left eye
	  //   cv::Point3f pupil_cornea_vectorL = cv::Point3f(resL->pupilCenter3D - resL->pupilCenter3D);
	  //   cv::Mat gaze_vec_in_augL = R_world2aug * cv::Mat(pupil_cornea_vectorL) / norm(pupil_cornea_vectorL);
	  //   cv::Mat gaze_vec_in_aug_corrL = K9_left * gaze_vec_in_augL;
	  //   cv::Mat gaze_vec_in_worldL = R_aug2world * gaze_vec_in_aug_corrL;

	  //   //gaze_vectors.push_back(cv::Point3d(gaze_vec_in_world.at<float>(0), gaze_vec_in_world.at<float>(1), gaze_vec_in_world.at<float>(2)));
	  //   //gaze_vectors.push_back(cv::Point3d(gaze_vec_in_world2.at<float>(0), gaze_vec_in_world2.at<float>(1), gaze_vec_in_world2.at<float>(2)));
	  //   gazeVectorR = cv::Point3d(gaze_vec_in_worldR.at<float>(0), gaze_vec_in_worldR.at<float>(1), gaze_vec_in_worldR.at<float>(2));
	  //   gazeVectorL = cv::Point3d(gaze_vec_in_worldR.at<float>(0), gaze_vec_in_worldR.at<float>(1), gaze_vec_in_worldR.at<float>(2));

	  // }
	  // else {
	    //gaze_vectors.push_back(K9_times_gaze_dir_point);
	  // gazeVectorR = resR->gazeDirectionVector;  // nää oli turhia... t: Miika
	  // gazeVectorL = resR->gazeDirectionVector;
	    //}

	  int blinking = 0;
	  if (resR->score + resL->score > 0.7) {  // good scores --> not blinking
	    blinking = 0;
	  }

	  double gaze_dist = DEFAULT_GAZE_DISTANCE; //TODO: this should probably be instantiated elsewhere for control

	  //gazepoint_ecam?, tarvitaanko best_eye_ind?
	  // Check if features of both eyes are ok; otherwise, use only "better" eye or don't use either
	  bool using_both_eyes = 1;
	  cv::Point3d gazePoint_ecam;
	  int best_eye_ind = 0; //0 = right, 1 = left
	  double best_score = resL->score;
	  if (abs(resR->score - resL->score) > 0.2) {
	    using_both_eyes = 0;
	    if (resR->score >= resL->score) { // added equality
	      best_eye_ind = 0; best_score = resR->score;
	      //gazePoint_ecam = resR->corneaCenter3D + gaze_dist * resR->gazeDirectionVector;  t: Miika
	      gazePoint_ecam = resR->corneaCenter3D + gaze_dist * gazeVectorR;
	    }
	    else{
	      best_eye_ind = 1; best_score = resL->score;
	      //gazePoint_ecam = resL->corneaCenter3D + gaze_dist * resL->gazeDirectionVector;
	      gazePoint_ecam = resL->corneaCenter3D + gaze_dist * gazeVectorL;
	    }
	  }

	  //TODO: not quite sure about this when gaze target is to the far left/right?
	  if (std::abs((resR->pupilEllipse.size.height - resL->pupilEllipse.size.height)) > imgSize.y*0.1 ||
	      std::abs((resR->pupilEllipse.size.width - resL->pupilEllipse.size.width)) > imgSize.x*0.1) {
	    //cout << "Huge difference in pupil sizes!" << endl;
	    if (resR->pupilEllipse.size.height > resL->pupilEllipse.size.height) { // assume that the smaller pupil is ok --> right eye
	      if (using_both_eyes) {
		using_both_eyes = 0;
		best_eye_ind = 0;
	      }
	      else {
		if (best_eye_ind = 1) { //if pupilsize would be equal, this will discard, can the above be '>='?
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

	  if (resR->corneaCenter3D.z < 0) {
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

	  if (using_both_eyes) {  // two cameras (and using both eyes)

	    cv::Point3d cr = resR->corneaCenter3D; // Right cornea center
	    cv::Point3d cl = resL->corneaCenter3D;  // Left cornea center
	    // cv::Point3d gr = resR->gazeDirectionVector;// / norm(resR->gazeDirectionVector);  // Right gaze vector (must be normalized) (it was t: Miika)
	    // cv::Point3d gl = resL->gazeDirectionVector;// / norm(resL->gazeDirectionVector);   // Left gaze vector (must be normalized)
	    cv::Point3d gr = gazeVectorR;
	    cv::Point3d gl = gazeVectorL;

	    // compute the gaze point as the 3d point with smallest square distance to the two gaze vectors. Results in noisy estimates, especially when looking further away.
	    // double gaze_dist_left2 = 1 / (1 - gr.dot(gl) * gl.dot(gr)) * (cr.dot(gl) - cl.dot(gl) + gr.dot(gl)*cl.dot(gr) - gr.dot(gl) * cr.dot(gr));  // "old" method, noisy...
	    // double gaze_dist_right2 = cl.dot(gr) - cr.dot(gr) + gaze_dist_left2*gl.dot(gr);

	    // Here the left and right gaze vectors are taken to be equally long which length is computed from simple trigonometry. This approach results in more robust estimate.
	    // TODO: where "robust" means "stable, but not quite correct?"
	    double gaze_dist_left = 0.5*norm(cr - cl) / sin(0.5*acos(gl.dot(gr)));
	    double gaze_dist_right = gaze_dist_left;

	    //KL T�� TOIMII VAAN SILL� OLETUKSELLA ETT� KATSE ON SILMIEN PUOLIV�LISS�? VARMASTI RAUHOITTAA TULOSTA, MUTTA EI SE OIKEIN OLE?!
	    //TODO: this is now in right eye coordinates, right? -> use of A_r2s below
	    gazePoint_ecam = (cv::Point3d(cl + gaze_dist_left*gl) + cv::Point3d(cr + gaze_dist_right*gr)) / 2;  // The computed gaze point is in the right (user's right) eye cam coordinates
	  }


	  double pog_x, pog_y;

	  //				if (!USE_REALSENSE) { TODO: for now, we don't include this
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

	  cv::UMat* sceneImage = frame->getImg(FrameSrc::SCENE);  // moved this here t: Miika
	  cv::flip(*sceneImage, *sceneImage, 1);    // The scene camera is upside down (t: Miika)
	  cv::flip(*sceneImage, *sceneImage, 0); 

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

	  /*
	    if (USE_FOVEATED_SMOOTHING) {  // voiko tän poistaa? t: Miika
	    sceneImage.convertTo(sceneImage, CV_32FC3);
	    Mat sceneImage_blur;
	    GaussianBlur(sceneImage, sceneImage_blur, Size(21, 21), 51);
	    float sigma2 = 10000;
	    for (int i = 0; i<Svert; i++) {
	    for (int j = 0; j<Shori; j++) {
	    for (int ch = 0; ch<3; ch++) {
	    float weight = exp(-((i - pog_scam.y)*(i - pog_scam.y) + (j - pog_scam.x)*(j - pog_scam.x)) / (2.0*sigma2));
	    sceneImage.at<Vec3f>(i, j)[ch] = float(weight * float(sceneImage.at<Vec3f>(i, j)[ch]) + \
	    (1 - weight) * (1 / 3.0 * (float(sceneImage_blur.at<Vec3f>(i, j)[0]) + float(sceneImage_blur.at<Vec3f>(i, j)[1]) + float(sceneImage_blur.at<Vec3f>(i, j)[2]))));
	    }
	    }
	    }
	    sceneImage.convertTo(sceneImage, CV_8UC3);
	    }

	  */

	  //std::cout << "FRAME_NR: " << frame->getNumber() << std::endl;  // miikan poisto

	  //cv::UMat* sceneImage = frame->getImg(FrameSrc::SCENE);  // moved upper...


	  // Draw the pog in the scene image. Adjust its size according to the glint fit score (of the "last" camera!).
	  cv::circle(*sceneImage, pog_scam, 35 * (1 - best_score) + 5, cv::Scalar(0, 0, 250), 7, 8);
	  
	  if (blinking) {  // TODO: Why it never glinks? 
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

	  //	char str[99]; sprintf(str, "%.2f", sigmoid);
	  //	cv::putText(*sceneImage, str, cv::Point2d(400, 50), CV_FONT_HERSHEY_PLAIN, 3, CV_RGB(250, 0, 100), 2);

	  //cv::UMat* temp = frame->getImg(FrameSrc::SCENE);
	  //std::string msg = "(Not) Analyzing Scene";
	  //cv::putText(temp->getMat(cv::ACCESS_WRITE), msg, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);

	  //TEST
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

	  // msecs frameTime = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
	  // std::cout << frameTime.count() << std::endl;  // miikan kellotus
			
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
