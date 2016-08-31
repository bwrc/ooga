//#pragma once
#ifndef VIDEOWRITER_NOTHREAD_H
#define VIDEOWRITER_NOTHREAD_H

#include "FrameBinocular.h"

//#include "boost/date_time/posix_time/posix_time.hpp"
//#include <boost/timer/timer.hpp>
#include <memory>
#include <chrono>
#include "SG_common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> //CV_FOURCC
#include <opencv2/videoio.hpp>
#include <assert.h>

// for notes about performance, see https://aaka.sh/patel/2013/06/28/live-video-webcam-recording-with-opencv/
// Apparently, VideoWriter needs to be told to write the frames every 1/FPS secs, otherwise it will stall. 
// in this non-threaded version, the tempo is actually driven by the calling participant (VideoIOHandler)

class VideoWriterNoThread
{
public:
	VideoWriterNoThread(std::string recordingName, int fps);
	~VideoWriterNoThread();

	bool setupVideoFile(FrameSrc src, std::string fn);

	void closeVideoFile(FrameSrc src);

	//void updateframe(TBinocularFrame* f);
	void updateframe(std::shared_ptr<TBinocularFrame> f);
	void write();

private:
	cv::VideoWriter *eyeLeftWriter;
	cv::VideoWriter *eyeRightWriter;
	cv::VideoWriter *sceneWriter;

	TBinocularFrame* currentframe;

	msecs zerotime;

	int fps = 30;

	bool frameUpdated = false;
};

#endif