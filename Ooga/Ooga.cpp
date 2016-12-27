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

#include "../utils/OOGUI.h"
#include "oogaConstants.h"
#include "SG_common.h"


void ModeCallBack(RunningModes mode, bool value){
	switch (mode){
	case(OOGA_MODE_CALIBRATE) :
		std::cout << "CALIBRATE" << std::endl;
		break;
	case (OOGA_MODE_RUNNING) :
		std::cout << "RUN" << std::endl;

		break;
	case (OOGA_MODE_PAUSED) :
		std::cout << "PAUSE" << std::endl;
		break;
	case (OOGA_MODE_RECORDING) :
		std::cout << "RECORD" << std::endl;
		break;
	}
}

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

  //setup video handler
  //TODO: read outputfilenames from config
  VideoIOHandler* grabber = new VideoIOHandler(processQueue, std::string("../videos/test"));
  grabber->setSaveState(true);

  FrameProcessor* processor = new FrameProcessor(processQueue, visualizationQueue, settings);
  std::cout << "PROCESSOR CREATED" << std::endl;

  //setup cameras
  //The cams vector should be populated in the order:
  // [0] : right, [1] : left, [2] : scene
  // now scene, left, right...?
  std::vector<cv::VideoCapture *> cams;
  std::vector<int> flipCodes;
  for (int i = 0; i < 3; i++){
    cams.push_back(new cv::VideoCapture());
	//should probably iterate through children of xml file cameras, but for now, assign indexes:

	//RIGHT EYE CAM
	if (i == 0){
		if (settings->eyeRightCam.type == 0){ //camera
			if (!cams[i]->open(settings->eyeRightCam.feedNumber)){
				std::cout << "Could not open camera " << settings->eyeRightCam.feedNumber << " for right eye feed!" << std::endl;
				break; //or exit
			}
		}
		else { //video file
			if (!cams[i]->open(settings->eyeRightCam.fileName)){
				std::cout << "Could not open file " << settings->eyeRightCam.fileName << " for right eye feed!" << std::endl;
				break; //or exit
			}
		}
		flipCodes.push_back(settings->eyeRightCam.flip);
	}
	//LEFT EYE CAM
	if (i == 1){
		if (settings->eyeLeftCam.type == 0){ //camera
			if (!cams[i]->open(settings->eyeLeftCam.feedNumber)){
				std::cout << "Could not open camera " << settings->eyeLeftCam.feedNumber << " for left eye feed!" << std::endl;
				break; //or exit
			}
		}
		else { //video file
			if (!cams[i]->open(settings->eyeLeftCam.fileName)){
				std::cout << "Could not open file " << settings->eyeLeftCam.fileName << " for left eye feed!" << std::endl;
				break; //or exit
			}
		}
		flipCodes.push_back(settings->eyeLeftCam.flip);
	}
	//SCENE CAM
	if (i == 2){
		if (settings->sceneCam.type == 0){ //camera
			if (!cams[i]->open(settings->sceneCam.feedNumber)){
				std::cout << "Could not open camera " << settings->sceneCam.feedNumber << " for scene feed!" << std::endl;
				break; //or exit
			}
		}
		else { //video file
			if (!cams[i]->open(settings->sceneCam.fileName)){
				std::cout << "Could not open file " << settings->sceneCam.fileName << " for scene feed!" << std::endl;
				break; //or exit
			}
		}
		flipCodes.push_back(settings->sceneCam.flip);
	}
	////if (!cams[i]->open("../videos/sg01_cam" + std::to_string(i + 1) + ".mjpg")) {
    ////  std::cout << "Could not open video file" << std::endl;
    ////  break;
    ////}

    cams[i]->set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cams[i]->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    //TODO: fuckety-foo - all of a sudden the compiler IN RELEASE MODE optimizes this out, probably due to
    //the enum(i) -> which might be undefined (while it shouldn't)
    int ret = grabber->AddCamera(FrameSrc(i), cams[i], flipCodes[i]);
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
/*  std::string wn0 = "cam0 | scene";
  std::string wn1 = "cam1 | left";
  std::string wn2 = "cam2 | right";

  cv::namedWindow(wn0, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
  cv::namedWindow(wn1, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
  cv::namedWindow(wn2, cv::WINDOW_NORMAL);// WINDOW_OPENGL);
  cv::moveWindow(wn2, 10, 500);
  cv::moveWindow(wn1, 650, 500);
  cv::moveWindow(wn0, 330, 50);
*/

  OOGUI* oogui = new OOGUI();
  oogui->Initialize();
  oogui->SetCallBackFunction( ModeCallBack );
/*=======
  cv::namedWindow(wn0, cv::WINDOW_AUTOSIZE);// WINDOW_OPENGL);
  cv::namedWindow(wn1, cv::WINDOW_AUTOSIZE);// WINDOW_OPENGL);
  cv::namedWindow(wn2, cv::WINDOW_AUTOSIZE);// WINDOW_OPENGL);
  cv::moveWindow(wn2, 110, 500); // Lisäsin 100 x-suuntaan t: Miika
  cv::moveWindow(wn1, 750, 500);
  cv::moveWindow(wn0, 430, 50);
*/
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

///	  cv::UMat *eyeImgL = myframe->getImg(FrameSrc::EYE_L);
///	  cv::UMat *eyeImgR = myframe->getImg(FrameSrc::EYE_R);
///	  cv::UMat *sceneImg = myframe->getImg(FrameSrc::SCENE);

	  //here, read tracking results and draw on overlays
	  //the tracker should not modify the frame

	//  cv::imshow(wn1, *eyeImgL);
	//  cv::imshow(wn2, *eyeImgR);
	//  cv::imshow(wn0, *sceneImg);

	///  std::vector<cv::Mat> imgs;
	///  imgs.push_back(eyeImgR->getMat(cv::ACCESS_READ));
	///  imgs.push_back(eyeImgL->getMat(cv::ACCESS_READ));
	///  imgs.push_back(sceneImg->getMat(cv::ACCESS_READ));
	///  imgs.push_back(sceneImg->getMat(cv::ACCESS_READ));

	///  oogui->pushFrame(imgs);

		oogui->pushFrame(*myframe);

	  counter++;
	  //std::cout << counter << std::endl;
	  //if (counter==10) exit(8);

	  //delete myframe;
	  myframe.reset();
	}

	//hrclock::time_point _start = hrclock::now();
	//msecs duration = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
	//std::cout << "FRAME DRAW: " << duration.count() << std::endl;

	//TODO sync this to 30Hz -> wait for 30 - measured_processing_time
/*	int key = cv::waitKey(1);

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
	  	//		case 49: //1
				//grabber->setFrameDurationTest(--framedur);
				//break;
				//case 50: //2
				//grabber->setFrameDurationTest(++framedur);
				//break;

	}
	*/

	oogui->update();

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
