// SmarterTracker.cpp : Defines the entry point for the console application.
//

//general todo's:
// - replace pragmas with ifndefs

//to "fix" the lack of gsl:
// linker->input->exclude specific /noinclude MSVCRTD.lib
//find gsl test -commented stuff and correct
//include corneacomputer .h .cpp to the project

//note: GSL apparently can't be compiled on VS2015
//see http://stackoverflow.com/questions/30412951/unresolved-external-symbol-imp-fprintf-and-imp-iob-func-sdl2
//-> answer about M$ -> added legacy_stdio_definitions.lib to project properties (both debug&release)
//-> still doesn't work?

#include "stdafx.h"
#include <iostream>

#include "FrameBinocular.h"
#include "VideoIOHandler.h"
#include "FrameProcessor.h"
#include "Settings.h"
#include "FrameRateLimiter.h"
//#include "Settings_noboost.h"

//these for the mc queue (it's not the bottleneck, so roll back to KL solution to reduce dependencies?)
#include "../utils/atomicops.h"
#include "../utils/readerwriterqueue.h"
namespace mc = moodycamel;

// MAIN
// to allow unicode on Win platforms, select main() VS-recommended formulation, else use C++ standard main() for multiplatform
#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[])
#else
int main(int argc, char** argv)
#endif
{

	/* SET PROGRAM OPTIONS, PATHS ETC.======================================================================*/

	//disable OpenCL to avoid the stutters
	putenv("OPENCV_OPENCL_RUNTIME=qqq");
	//this should enable OPENCL?
	//putenv("OPENCV_OPENCL_RUNTIME=");

	//set number of threads available for multithreading OpenCV functions
	//std::cout << "before: " << cv::getNumThreads() << std::endl;
	cv::setNumThreads(8);
	//std::cout << "after: " << cv::getNumThreads() << std::endl;
	//std::cin.ignore();

	//read settings
	TSettings *settings = new TSettings();
	//Settings_noboost *settings = new Settings_noboost();
	if (!settings->processCommandLine(argc, argv)){
		std::cout << "invalid command line -> exit" << std::endl;
		return 0;
	}
	settings->loadSettings(settings->configFile);

	/* SET UP queues, threads, etc.  =======================================================================*/

	mc::ReaderWriterQueue<TBinocularFrame *> processQueue(100);
	mc::ReaderWriterQueue<TBinocularFrame *> visualizationQueue(100);

	//rate limiter
	FrameRateLimiter* limiter = new FrameRateLimiter();

	//setup video handler
	VideoIOHandler* grabber = new VideoIOHandler(&processQueue, std::string("..\\videos\\test"));
	grabber->setSaveState(true); 

	FrameProcessor* processor = new FrameProcessor(&processQueue, &visualizationQueue);

	//setup cameras
	//TODO:
	//this only opens a set of video files for testing, replace with configged opening of cameras!
	std::vector<cv::VideoCapture *> cams;
	for (int i = 0; i < 3; i++){
		cams.push_back(new cv::VideoCapture());
		//		cams[i]->open("../videos/video" + std::to_string(i) + ".avi");
		if (!cams[i]->open("../videos/cam" + std::to_string(i + 1) + ".mjpg")) {
			std::cout << "Could not open video file" << std::endl;
		}

		cams[i]->set(cv::CAP_PROP_FRAME_WIDTH, 640);
		cams[i]->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

		grabber->AddCamera(FrameSrc(i), cams[i]);
		// (int)FrameSrc(i)]); 
		//todo: addcamera doesn't use the frame identifier, just pushes them in order called, how should this work to get the right feed to the right tracker?
	}

	//setup cv windows
	std::string wn0 = "cam0";
	std::string wn1 = "cam1";
	std::string wn2 = "cam2";
	cv::namedWindow(wn0, cv::WINDOW_OPENGL);
	cv::namedWindow(wn1, cv::WINDOW_OPENGL);
	cv::namedWindow(wn2, cv::WINDOW_OPENGL);
	cv::moveWindow(wn2, 10,500);
	cv::moveWindow(wn1, 650, 500);
	cv::moveWindow(wn0, 330,50);

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
			TBinocularFrame* myframe;
			if (visualizationQueue.try_dequeue(myframe)){

				cv::UMat *eyeImgL = myframe->getImg(FrameSrc::EYE_L);
				cv::UMat *eyeImgR = myframe->getImg(FrameSrc::EYE_R); 
				cv::UMat *sceneImg = myframe->getImg(FrameSrc::SCENE);

				//here, read tracking results and draw on overlays
				//the tracker should not modify the frame

				cv::imshow(wn1, *eyeImgL);
				cv::imshow(wn2, *eyeImgR);
				cv::imshow(wn0, *sceneImg);

				counter++;

				delete myframe;
			}
			//TODO: sync this to 30Hz -> wait for 30 - measured_processing_time
//			grabber->pause(); //miikabugi 
			int key = cv::waitKey(1);
//			grabber->pause(); //miikabugi 
			//miikabugi int key = cv::waitKey(1);
			switch (key){
				//see http://www.expandinghead.net/keycode.html
			case 27: //ESC
				stopThisNonsense = true;
				break;
			case 32: //SPACE
				grabber->pause(); //toggle
				break;
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
	//TODO: should grabber close opened files, would that create valid containers?

	// CLEANUP ---------------------
	for (auto &cam : cams){
		delete cam;
	}
	cams.clear();

	delete grabber;

	delete limiter;

	return 0;
}

