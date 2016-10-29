/*
// This file is part of the OOGA - Open-source Gazetracker Application
// Copyright 2016
// Kristian Lukander <kristian.lukander@ttl.fi>,
// Miika Toivanen <miika.toivanen@ttl.fi>
// Finnish Institute of Occupational Health, Helsinki, Finland
//
// Please see the file LICENSE for details.*/

/* general notes on this file
//
//to "fix" the lack of gsl:
// linker->input->exclude specific /noinclude MSVCRTD.lib
//find gsl test -commented stuff and correct
//include corneacomputer .h .cpp to the project

//note: GSL apparently can't be compiled on VS2015
//see http://stackoverflow.com/questions/30412951/unresolved-external-symbol-imp-fprintf-and-imp-iob-func-sdl2


NOTE: For Visual Studio, must add the following for debugging properties (with proper user paths):
command line: --config config\config.xml
environment: C:\Dev\gsl\x86\lib;C:\Dev\opencv3\bin\install\x86\vc12\bin;C:\Dev\boost\lib;C:\Dev\tbb44_20160526oss\bin\ia32\vc14 
(TBB optional on OpenCV)

// ===========================================================================================================*/

#ifdef WIN32
	#include "stdafx.h"
#endif

#include <iostream>

#include "FrameBinocular.h"
#include "VideoIOHandler.h"
#include "FrameProcessor.h"
#include "Settings.h"
#include "FrameRateLimiter.h"
#include "../utils/concurrent_queue.h"

// MAIN
// to allow unicode on Win platforms, select main() VS-recommended formulation, else use C++ standard main() for multiplatform
#ifdef _WIN32
	int _tmain(int argc, _TCHAR* argv[])
#else
	int main(int argc, char** argv)
#endif
{
	/* SET PROGRAM OPTIONS, PATHS ETC.======================================================================*/

	// TODO these should be configurable without compiling -> read settings first
	//disable OpenCL to avoid the stutters
	putenv("OPENCV_OPENCL_RUNTIME=qqq");
	//this should enable OPENCL
	//putenv("OPENCV_OPENCL_RUNTIME=");

	//set number of threads available for multithreading OpenCV functions
	cv::setNumThreads(8);

	//read settings
	TSettings *settings = new TSettings();

	// TODO Implement error-checking for badly formatted config files
	if (!settings->processCommandLine(argc, argv)){
		std::cout << "invalid command line -> exit" << std::endl;
		return 0;
	}
	settings->loadSettings(settings->configFile);

	/* SET UP queues, threads, etc.  =======================================================================*/
	//initialize queues
//	concurrent_queue<std::shared_ptr<TBinocularFrame>> *processQueue;
//	processQueue = new concurrent_queue<std::shared_ptr<TBinocularFrame>>;
	BalancingQueue<std::shared_ptr<TBinocularFrame>> *processQueue;
	processQueue = new BalancingQueue<std::shared_ptr<TBinocularFrame>>;

//	concurrent_queue<std::shared_ptr<TBinocularFrame>> *visualizationQueue;
//	visualizationQueue = new concurrent_queue<std::shared_ptr<TBinocularFrame>>;
	BalancingQueue<std::shared_ptr<TBinocularFrame>> *visualizationQueue;
	visualizationQueue = new BalancingQueue<std::shared_ptr<TBinocularFrame>>;

	//rate limiter
//	FrameRateLimiter* limiter = new FrameRateLimiter(); <- included in the queues

std::cout << "BEFORE VIDEOIO" << std::endl;

	//setup video handler
	VideoIOHandler* grabber = new VideoIOHandler(processQueue, std::string("../videos/test"));
	grabber->setSaveState(true);
	std::cout << "VIDEOIO OK" << std::endl;

	FrameProcessor* processor = new FrameProcessor(processQueue, visualizationQueue);
	std::cout << "PROCESSOR CREATED" << std::endl;

	//setup cameras
	//TODO this only opens a set of video files for testing, replace with configged opening of cameras!
	std::vector<cv::VideoCapture *> cams;
	for (int i = 0; i < 3; i++){
		cams.push_back(new cv::VideoCapture());
		//		cams[i]->open("../videos/video" + std::to_string(i) + ".avi");
		if (!cams[i]->open("../videos/cam" + std::to_string(i + 1) + ".mjpg")) {
			std::cout << "Could not open video file" << std::endl;
			break;
		}

		cams[i]->set(cv::CAP_PROP_FRAME_WIDTH, 640);
		cams[i]->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

		//TODO: fuckety-foo - all of a sudden the compiler IN RELEASE MODE optimizes this out, probably due to
		//the enum(i) -> which might be undefined (while it shouldn't)
		int ret = grabber->AddCamera(FrameSrc(i), cams[i]);
		std::cout << ret;
/*		switch (i){
		case 0:
			grabber->AddCamera(FrameSrc::SCENE, cams.at(i));// [i]);
			break;
		case 1:
			grabber->AddCamera(FrameSrc::EYE_L, cams[i]);
			break;
		case 2:
			grabber->AddCamera(FrameSrc::EYE_R, cams[i]);
			break;
		}
*/
		// (int)FrameSrc(i)]);
		// TODO addcamera doesn't use the frame identifier, just pushes them in order called, how should this work to get the right feed to the right tracker?
	}
	std::cout << "AFTER CAM ADD" << std::endl;

	//setup cv windows
	std::string wn0 = "cam0";
	std::string wn1 = "cam1";
	std::string wn2 = "cam2";
	cv::namedWindow(wn0, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
	cv::namedWindow(wn1, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
	cv::namedWindow(wn2, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
	cv::moveWindow(wn2, 10, 500);
	cv::moveWindow(wn1, 650, 500);
	cv::moveWindow(wn0, 330, 50);

	bool stopThisNonsense = false;
	bool displayStats = true;

	int counter = 0;

	//start the threaded grabber & processor
	grabber->start();
	processor->start();

	//MAIN LOOP visualizing the results
	while (!stopThisNonsense){

		try
		{
			auto myframe = std::shared_ptr<TBinocularFrame>(nullptr);
			//if (visualizationQueue.try_dequeue(myframe)){
				if (visualizationQueue->try_pop(myframe)){

				cv::UMat *eyeImgL = myframe->getImg(FrameSrc::EYE_L);
				cv::UMat *eyeImgR = myframe->getImg(FrameSrc::EYE_R);
				cv::UMat *sceneImg = myframe->getImg(FrameSrc::SCENE);

				//here, read tracking results and draw on overlays
				//the tracker should not modify the frame

				cv::imshow(wn1, *eyeImgL);
				cv::imshow(wn2, *eyeImgR);
				cv::imshow(wn0, *sceneImg);

				counter++;

				//delete myframe;
				myframe.reset();
			}

			//TODO sync this to 30Hz -> wait for 30 - measured_processing_time
			int key = cv::waitKey(1);
			switch (key){
				//see http://www.expandinghead.net/keycode.html
			case 27: //ESC
				stopThisNonsense = true;
				break;
			case 32: //SPACE
				grabber->pause(); //toggle
				break;
			case 110: //n
				grabber->grabone();
				/*			case 49: //1
				grabber->setFrameDurationTest(--framedur);
				break;
				case 50: //2
				grabber->setFrameDurationTest(++framedur);
				break;
				*/
			}
		}
		catch (int e) {
			std::cout << "error in main: " << e << std::endl;
		}
	}

	grabber->stop();
	// TODO should grabber close opened files, would that create valid containers?

	// CLEANUP ---------------------
	for (auto &cam : cams){
		delete cam;
	}
	cams.clear();

	delete grabber;

	//delete limiter;

	return 0;
}
