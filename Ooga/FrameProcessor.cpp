#include "FrameProcessor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <thread>

FrameProcessor::FrameProcessor(concurrent_queue<std::shared_ptr<TBinocularFrame>>* in,
								concurrent_queue<std::shared_ptr<TBinocularFrame>>* out)
{
	this->qIn = in;
	this->qOut = out;

	etLeft = new EyeTracker();
	etRight = new EyeTracker();

	//TODO Move hard-coded files to settings
	std::string CM_fn_left = "../calibration/file_CM_left";
	std::string CM_fn_right = "../calibration/file_CM_right";
	std::string glintmodel_fn = "../calibration/glint_model.yaml";

	etLeft->InitAndConfigure( FrameSrc::EYE_L, CM_fn_left, glintmodel_fn ); 
	etRight->InitAndConfigure(FrameSrc::EYE_R, CM_fn_right, glintmodel_fn ); 

	//st = new TSceneTracker();
	//st->SetCamFeedSource(FrameSrc::SCENE);
	//st->InitAndConfigure(); // Doesn't read settings now. t: Miika

	pauseWorking = false;

	//ptimer = new TPerformanceTimer();

}

FrameProcessor::~FrameProcessor()
{
	if (etLeft != nullptr){
		delete etLeft;
	}
	if (etLeft != nullptr) {
		delete etRight;
	}
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

			//_start = hrclock::now();

			//TBinocularFrame* frame;
			auto frame = std::shared_ptr<TBinocularFrame>(nullptr);

			//if (qIn->try_dequeue(frame)){
			if (qIn->try_pop(frame)){

				my_mtx.lock();

				cv::Point3d pupilCenter3DL, corneaCenter3DL;
				cv::Point3d pupilCenter3DR, corneaCenter3DR;
				//				cv::Mat K9_matrix = et->K9_matrix;
//				Eigen::Matrix4d A = st->A_matrix;
//				Camera sceneCam = st->sceneCam;

//				ptimer->start();
				//this now runs the processor in this thread
				//et->Process(frame, brightestEye);

				TTrackingResult resL;
				TTrackingResult resR;

				//runs left in it's own thread and right here
//DEBUGMONOCULAR				
				std::thread thr_eL = std::thread(&EyeTracker::Process, etLeft, frame->getImg(FrameSrc::EYE_L), resL, pupilCenter3DL, corneaCenter3DL);
				//etLeft->Process(frame->getImg(FrameSrc::EYE_L), resL, pupilCenter3DL, corneaCenter3DL);

				//std::thread thr_eR = std::thread(&EyeTracker::Process, etRight, frame->getImg(FrameSrc::EYE_R), resR, pupilCenter3DL, corneaCenter3DL);
				etRight->Process(frame->getImg(FrameSrc::EYE_R), resR, pupilCenter3DR, corneaCenter3DR);
				
				/* HUOMAA ETTÄ etRIGHT KÄYTTÄÄ NYT EYE_L-kuvaa, enumerointi menee väärin*/

//DEBUGMONOCULAR
				thr_eL.join();
				//thr_eR.join();

//cv::UMat* f = frame->getImg(FrameSrc::EYE_L);
				//et->Process(*f, res, pupilCenter3D, corneaCenter3D);

				// st->Process(frame, brightestScene);  // Don't process scene image now

				//here, to run the processing in another thread - right now with OpenCV3.0 results in strange sync errors probably due to sync problems in GPU pipeline (OpenCV internal problem)
				//eyeThread = boost::thread(&TEyeTracker::Process, et, frame, &pupilLoc);
				//eyeThread = boost::thread(&TEyeTracker::Process, et, eyeMat, pf);

				//sceneThread = boost::thread(&TBaseTracker::Process, st, frame);
				//sceneThread = boost::thread(&TSceneTracker::Process, st, sceneMat);

				//wait for threads to do their stuff
				//eyeThread.join();
				//sceneThread.join();

//				std::cout << "proctime: " << ptimer->elapsed_ms() << std::endl;

				// ----- Map the pupil-corneal distance vector to scene camera Vector3d
				//double gaze_dist = 1.0;  // The gaze distance
/*				double gaze_dist = st->gaze_dist;
				double inv_norm = 1.0 / double(cv::norm(pupilCenter3D - corneaCenter3D));
				cv::Point3f gaze_dir = (pupilCenter3D - corneaCenter3D) * inv_norm;
				cv::Mat gaze_dir_mat(gaze_dir);
				cv::Mat K9_times_gaze_dir_mat = K9_matrix * gaze_dir_mat;
				cv::Point3d K9_times_gaze_dir_point(K9_times_gaze_dir_mat.at<float>(0), K9_times_gaze_dir_mat.at<float>(1), K9_times_gaze_dir_mat.at<float>(2));
				cv::Point3d gazePoint_ecam = corneaCenter3D + gaze_dist * K9_times_gaze_dir_point;

				Eigen::Vector4d gazePoint_ecam_aug(gazePoint_ecam.x, gazePoint_ecam.y, gazePoint_ecam.z, 1);
				Eigen::Vector4d gazePoint_scam_aug = A * gazePoint_ecam_aug;
				cv::Point3d gazePoint_scam = cv::Point3d(gazePoint_scam_aug(0), gazePoint_scam_aug(1), gazePoint_scam_aug(2));
				inv_norm = 1 / norm(gazePoint_scam);
				cv::Point3d gazePoint_scam_norm = gazePoint_scam * inv_norm;

				double pog_x, pog_y;  // These are THE result: point-of-gaze in scene camera coordinates
				sceneCam.worldToPix(gazePoint_scam_norm, &pog_x, &pog_y);
				cv::Point2d pog_scam(pog_x, pog_y);

				st->DirtyDrawer(frame, pog_scam);  // A dirty drawer; remove this from the final version!

				// do joint work based on the results
				TTrackingResult tr;
				tr.val = 1;
				frame->setTrackingResult(tr);

				// push to processed frames queue
				qOut->push(frame);

				// remove refs
				frame.reset();
*/
				cv::UMat* temp = frame->getImg(FrameSrc::SCENE);

				std::string msg = "(Not) Analyzing Scene";

				cv::putText(temp->getMat(cv::ACCESS_WRITE), msg, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);

				//qOut->enqueue(frame);
				qOut->push(frame);

				my_mtx.unlock();

			}
			msecs frameTime = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
			msecs waitTime = msecs(30) - frameTime;

			std::cout << "frametime: " << frameTime.count() << std::endl;

			if (waitTime > msecs(0)){
				std::this_thread::sleep_for(waitTime);
			}

		}
		catch (int e) {
			std::cout << "error in processor: " << e << std::endl;
		}

	}
	
}

