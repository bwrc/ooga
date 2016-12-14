#include "VideoIOHandler.h"

#include <iostream>
#include "SG_common.h"

//Constructor takes a pointer to the processing queue, and a filename for saving the videos
//VideoIOHandler::VideoIOHandler(concurrent_queue<std::shared_ptr<TBinocularFrame>>* procRWQ, std::string recordingName)
VideoIOHandler::VideoIOHandler(BalancingQueue<std::shared_ptr<TBinocularFrame>>* procRWQ, std::string recordingName)
{
	this->caps.clear();

	this->procRWQ = procRWQ;
	this->recName = recordingName;

	this->bSaveFrames = false;
	this->totalGrabbed = 0;

	zerotime = std::chrono::steady_clock::now();
	pauseGrabbing = false;

	writer = new VideoWriterNoThread(recordingName, 25);
	//this bugger's already threaded, skip creating another thread for writing
}

VideoIOHandler::~VideoIOHandler()
{
	delete writer;

	//vectors get automatically deallocated
	caps.clear();
}


/** Add a camera to grab
*	\param cap a pointer to a VideoCapture from the calling entity
*	\return the number of the camera assigned (needed for at least the video saver)
*/
int VideoIOHandler::AddCamera(FrameSrc f, cv::VideoCapture *cap)
{
	//todo: vector fills up regardless of FrameSrc. Ok, if in "right order"
	//should these just be three different caps?
	caps.push_back(cap);
	return caps.size();
}

/** GrabFramesInThread()
* grabs a frame from all of the cameras as synced as possible
* returns the number of frames grabbed
*/

void VideoIOHandler::grabone()
{
	grabsingleframe = true;
	if (!(this->pauseGrabbing)){ //if not already paused
		this->pause(); //this is a toggle, start would try to rerun m_thread
	}
}

void VideoIOHandler::GrabFramesInThread()
{
	hrclock::time_point _start = hrclock::now();
	TGrabStatistics gs;
	
	while (keepRunning){ // keep the thread running
		try{
			blockWhilePaused(); //enable pausing

			std::vector<bool> frameGrabbed;// (3); //avoid reallocating twice

			//first a quick round of grabbing to improve sync
			hrclock::time_point grabstart = hrclock::now(); 
			for (auto &cap : caps){
				if (cap->grab()){
					frameGrabbed.push_back(true);
					//todo: record diff time between cam caps and save to frame
				}
				else {
					frameGrabbed.push_back(false);
				}
			}
			gs.grabTime = std::chrono::duration_cast<msecs>(hrclock::now() - grabstart);

			//check that we got a grab for all cameras
			// is this overly complicated, just use int++?
			bool gotAllFrames = true;
			for (auto g : frameGrabbed){
				if (!g){ gotAllFrames = false; }
			}

			if (gotAllFrames){
				++totalGrabbed;

				//create a frame for holding image pairs & metadata 
				//TBinocularFrame* frame = new TBinocularFrame();
				auto frame = std::make_shared<TBinocularFrame>();

				frame->setNumber(totalGrabbed);

				//todo: is this good?
				int i = 0;
				cv::UMat *grab;
				for (auto &cap : caps){
					grab = frame->getImg(FrameSrc(i));
					cap->retrieve(*grab);
					++i;
				}

				gs.frameTime = std::chrono::duration_cast<msecs>(hrclock::now() - _start);
				frame->setTimestamp(gs.frameTime);// std::chrono::duration_cast<msecs>(hrclock::now() - _start));

				//boost::this_thread::interruption_point();
				// time performance and minus from 30 (hardcoded max framerate?! TODO: allow configurable?)
				//TODO: this is wrong too, saving below takes time?!
				//use msecs(30) - (now()-_grabstart) just before the wait below?
				gs.waitTime = msecs(30) - gs.frameTime;

//				TBinocularFrame* sframe;// = nullptr; //placeholder
				//frame has to be copied before pushing to queue so that it won't be manipulated!
				if (bSaveFrames){
					writer->updateframe(frame);
					//makes a deep copy of original frame for saving 
				}

				//push frame to queue
				//procRWQ->enqueue(frame);
//				procRWQ->push(frame);
				//instead of just pushing,
				//the balancing queue needs to block if it can't push 
				//(so as to not skip frames, which could be ok for realtime slow processing while saving all frames?)
/*simple				while (!(procRWQ->try_push(frame))){
					//waiting for the push to go through (the wait should be enough)
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
*/
				if (!(procRWQ->try_push(frame))){
					//sleep for the average frame processing time and then try until it gets through
					const unsigned long timeToSleep = static_cast<unsigned long>(procRWQ->getAverageConsumerTime());
					//std::cout << "************ IO waiting for: " << timeToSleep << std::endl;
					std::this_thread::sleep_for(std::chrono::milliseconds(timeToSleep));
					while (!(procRWQ->try_push(frame))){
						//waiting for the push to go through (the wait above should be enough)
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
					}
				}

				//non-threaded videowrite:
				//the other option is to have a separate worker thread, or a sep thread for writing
				//slow disk ops after enqueing the current frame for processing

				//TODO: for balancing queue this could be done while waiting?
				if (bSaveFrames){
					writer->write();
				}

			}
			else
			{
				int i = 0;
				//for (auto &g : frameGrabbed){
				for (auto g : frameGrabbed){
					if (!g){
						std::cout << "Frame " << i << " gone missing, can't grab" << std::endl;
					}
				}
			}

			if (grabsingleframe){ //just a single run through was requested
				grabsingleframe = false;
				this->pause();
			}

			if (gs.waitTime > msecs(0)){
				std::this_thread::sleep_for(gs.waitTime);
			}

		}
		catch (int e) {
			std::cout << "error in grabber: " << e << std::endl;
		}
	}

}

void VideoIOHandler::setSaveState(bool state)
{
	this->bSaveFrames = state;
}

void VideoIOHandler::blockWhilePaused()
{
	std::unique_lock<std::mutex> lock(pauseMutex);
	while (pauseGrabbing)
	{
		pauseChanged.wait(lock);
	}
}

void VideoIOHandler::start()
{
	zerotime = std::chrono::steady_clock::now();
	m_thread = std::thread(&VideoIOHandler::GrabFramesInThread, this);
}

void VideoIOHandler::pause()
{
	//boost::unique_lock<boost::mutex> lock(pauseMutex);
	std::unique_lock<std::mutex> lock(pauseMutex);
	pauseGrabbing = !pauseGrabbing; // this now functions as a toggle, rather than 'pause'

	pauseChanged.notify_all();
}

void VideoIOHandler::stop()
{
	//m_thread.interrupt(); //this is not implemented in std for stack allocation reasons, might have to resort back to boost?
	keepRunning = false;
	if (m_thread.joinable()){
		m_thread.join();
	}
}
