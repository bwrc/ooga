//#pragma once
#ifndef FRAMEPROCESSOR
#define FRAMEPROCESSOR

#include "FrameBinocular.h"
#include <memory> // for shared_ptr

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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
	FrameProcessor(concurrent_queue<std::shared_ptr<TBinocularFrame>>* in,
					concurrent_queue<std::shared_ptr<TBinocularFrame>>* out);
	~FrameProcessor();

	/**
	Starts up the FrameProcessor
	*/
	void start();
	void pause();
	void stop();

	void Process();

private:
	concurrent_queue<std::shared_ptr<TBinocularFrame>>* qIn;
	concurrent_queue<std::shared_ptr<TBinocularFrame>>* qOut;

	std::thread m_thread;
	std::mutex my_mtx;

	//boost::thread eyeThread;
	//boost::thread sceneThread;

	EyeTracker *etLeft;
	EyeTracker *etRight;
	//TSceneTracker *st;

	std::chrono::steady_clock::time_point zerotime;

	//TPerformanceTimer *ptimer;

	void blockWhilePaused();
	bool pauseWorking;
	std::atomic<bool> running;
	std::mutex pauseMutex;
	std::condition_variable pauseChanged;
};

#endif
