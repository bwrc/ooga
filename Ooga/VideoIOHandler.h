//#pragma once
#ifndef VIDEO_IO_HANDLER_H
#define VIDEO_IO_HANDLER_H

#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <string>

//#include <boost/thread/thread.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>

//writer #include "VideoWriter.h"

#include "FrameBinocular.h"
#include "VideoWriterNoThread.h"

#include "../utils/concurrent_queue.h"

class VideoIOHandler
{
public:
	//Constructor takes a pointer to the processing queue
	//VideoIOHandler(mc::ReaderWriterQueue<TBinocularFrame *>* procRWQ, std::string recordingName); //camel
	VideoIOHandler(concurrent_queue<std::shared_ptr<TBinocularFrame>>* procRWQ, std::string recordingName);

	~VideoIOHandler();

	/** Add a camera to grab
	*	\param cap a pointer to a VideoCapture from the calling entity
	*	\return the number of the camera assigned (needed for at least the video saver)
	*/
	int AddCamera(FrameSrc f, cv::VideoCapture *cap);

	/** GrabFramesInThread()
	* grabs a frame from all of the cameras as synced as possible
	* returns the number of frames grabbed
	*/
	void GrabFramesInThread();

	//threaded version
	void setSaveState(bool state);
	//non-threaded videowrite: 
	//void setSaveState(bool state, std::string eyeFileName, std::string sceneFileName); 

	//void pauseGrabber();// bool shouldPause);
	void start();
	void pause();
	void stop();

private:

	//boost::thread m_thread;
	std::thread m_thread;
	std::atomic<bool> keepRunning = true;

	std::string recName = "";

	//the processing queue where grabbed frames will be pushed
	//mc::ReaderWriterQueue<TBinocularFrame *>* procRWQ;
	concurrent_queue<std::shared_ptr<TBinocularFrame>>* procRWQ;

	std::vector<cv::VideoCapture *> caps;				// pointers to video capture devices 

	int64 totalGrabbed;
	//boost::posix_time::ptime zerotime;
	std::chrono::steady_clock::time_point zerotime;

	VideoWriterNoThread* writer;
	bool bSaveFrames;

	void blockWhilePaused();
	bool pauseGrabbing;
	std::mutex pauseMutex;
	std::condition_variable pauseChanged;
};

//#include "VideoWriterNoThread.h"
//#define BUFFER_LENGTH 32 TODO: give the queues a maximum size,decide  what happens if it runs full?

//todo:
//control camera number better?
//control frame saving from settings -> set the boolean

#endif
