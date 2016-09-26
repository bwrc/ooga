//#pragma once
#ifndef FRAMEPROCESSOR
#define FRAMEPROCESSOR

#include "FrameBinocular.h"
#include <memory> // for shared_ptr

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <opencv2/core/core.hpp>

#include "SG_common.h"
#include <chrono>

#include "..\utils\concurrent_queue.h"
#include "EyeTracker.h"


//#include "PerformanceTimer.h"

/** Frame Processor
*	Reads Frames from a queue, assigns individual camera grabs for threaded trackers, and pushes the resulting Processed Frame to the out queue.
*/

class FrameProcessor
{
public:
	//FrameProcessor(mc::ReaderWriterQueue<TBinocularFrame *>* in, mc::ReaderWriterQueue<TBinocularFrame *>* out);
//	FrameProcessor(concurrent_queue<std::shared_ptr<TBinocularFrame>>* in,
//					concurrent_queue<std::shared_ptr<TBinocularFrame>>* out);
	FrameProcessor(BalancingQueue<std::shared_ptr<TBinocularFrame>>* in,
		BalancingQueue<std::shared_ptr<TBinocularFrame>>* out);
	~FrameProcessor();

	/**
	Starts up the FrameProcessor
	*/
	void start();
	void pause();
	void stop();

	void Process();

private:
//	concurrent_queue<std::shared_ptr<TBinocularFrame>>* qIn;
//	concurrent_queue<std::shared_ptr<TBinocularFrame>>* qOut;
	BalancingQueue<std::shared_ptr<TBinocularFrame>>* qIn;
	BalancingQueue<std::shared_ptr<TBinocularFrame>>* qOut;

	std::thread m_thread;
	std::mutex my_mtx;

	//boost::thread eyeThread;
	//boost::thread sceneThread;

	Camera* eyeCamL;
	Camera* eyeCamR;
	Camera* sceneCam;

	EyeTracker *etLeft;
	EyeTracker *etRight;
	//TSceneTracker *st;

	//there's a strange name conflict affecting eigen and cv::Mat constructor calling,
	//resulting in "expected a type specifier" error with Eigen::MatrixXd A(4,4) and cv::Mat A(3,3,CV_32F)
	//while can't find, this seems to be ok:
	//Eigen::MatrixXd A = Eigen::MatrixXd(4, 4);  // This is the matrix that transforms (augmented) coordinates from the eye camera to the scene camera
	Eigen::MatrixXd A_l2s = Eigen::MatrixXd(4, 4);  // This is the matrix that transforms (augmented) coordinates from the eye camera to the scene camera
	Eigen::MatrixXd A_r2s = Eigen::MatrixXd(4, 4);  // This is the matrix that transforms (augmented) coordinates from the eye camera to the scene camera
	Eigen::MatrixXd A_l2r = Eigen::MatrixXd(4, 4);  // the transformation from left eye camera to the right eye camera

	cv::Point2d imgSize = cv::Point2d(640, 480);

	cv::Mat K9_left;
	cv::Mat K9_right;

	std::chrono::steady_clock::time_point zerotime;

	//TPerformanceTimer *ptimer;

	void blockWhilePaused();
	bool pauseWorking;
	std::atomic<bool> running;
	std::mutex pauseMutex;
	std::condition_variable pauseChanged;
};

#endif
