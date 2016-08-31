#include "VideoWriterNoThread.h"
#include <opencv2\highgui\highgui.hpp>
VideoWriterNoThread::VideoWriterNoThread(std::string recordingName, int fps)
{
	this->fps = fps;

	//todo: react to bools
	std::string  eyeLName = recordingName + "Left.avi";
	std::string  eyeRName = recordingName + "Right.avi";
	std::string  sceneName = recordingName + "Scene.avi";

	setupVideoFile(FrameSrc::EYE_L, eyeLName);
	setupVideoFile(FrameSrc::EYE_R, eyeRName);
	setupVideoFile(FrameSrc::SCENE, sceneName);

	int delayFound = 0;
	int totalDelay = 0;

	currentframe = new TBinocularFrame();
	frameUpdated = false;

}

VideoWriterNoThread::~VideoWriterNoThread()
{
	if (eyeLeftWriter->isOpened()){
		eyeLeftWriter->release();
	}
	delete eyeLeftWriter;

	if (eyeRightWriter->isOpened()){
		eyeRightWriter->release();
	}
	delete eyeRightWriter;

	if (sceneWriter->isOpened()){
		sceneWriter->release();
	}
	delete sceneWriter;

	delete currentframe;
}

bool VideoWriterNoThread::setupVideoFile(FrameSrc src, std::string fn)
{
	int fps = this->fps;
	int width = 640;
	int height = 480;

	bool retval = true;

	switch (src){
	case FrameSrc::EYE_L:
		eyeLeftWriter = new cv::VideoWriter(
			fn,
			CV_FOURCC('D', 'I', 'V', 'X'),
			fps,
			cv::Size(width, height),
			true); //should be false for bw?

		if (!eyeLeftWriter->isOpened()){
			retval = false;
		}
		break;
	case FrameSrc::EYE_R:
		eyeRightWriter = new cv::VideoWriter(
			fn,
			CV_FOURCC('D', 'I', 'V', 'X'),
			fps,
			cv::Size(width, height),
			true); 

		if (!eyeRightWriter->isOpened()){
			retval = false;
		}
		break;
	case FrameSrc::SCENE:
		sceneWriter = new cv::VideoWriter(
			fn,
			CV_FOURCC('D', 'I', 'V', 'X'),
			fps,
			cv::Size(width, height),
			true); 

		if (!sceneWriter->isOpened()){
			retval = false;
		}
		break;
	}

	return retval;
}


void VideoWriterNoThread::closeVideoFile(FrameSrc src)
{
	switch (src){
	case FrameSrc::EYE_L:
		if (eyeLeftWriter->isOpened()){
			eyeLeftWriter->release();
		}
		break;
	case FrameSrc::EYE_R:
		if (eyeRightWriter->isOpened()){
			eyeRightWriter->release();
		}
		break;
	case FrameSrc::SCENE:
		if (eyeRightWriter->isOpened()){
			eyeRightWriter->release();
		}
		break;
	}
}

//void VideoWriterNoThread::updateframe(TBinocularFrame* f){
	void VideoWriterNoThread::updateframe(std::shared_ptr<TBinocularFrame> f){

	//todo: copy stats
	//currentframe->setGrabbingStats(frame->setGrabbingStats());
	//+trackingresults
	cv::UMat *eyeLeftMat = currentframe->getImg(FrameSrc::EYE_L);
	cv::UMat *eyeRightMat = currentframe->getImg(FrameSrc::EYE_R);
	cv::UMat *sceneMat = currentframe->getImg(FrameSrc::SCENE);
	f->getImg(FrameSrc::EYE_L)->copyTo(*eyeLeftMat);
	f->getImg(FrameSrc::EYE_R)->copyTo(*eyeRightMat);
	f->getImg(FrameSrc::SCENE)->copyTo(*sceneMat);

	frameUpdated = true;
}

void VideoWriterNoThread::write()
{
	if (frameUpdated){
		//TODO: add better error condition handling
		assert(eyeLeftWriter->isOpened() && eyeRightWriter->isOpened() && sceneWriter->isOpened());

		cv::UMat *eyeLeftMat = currentframe->getImg(FrameSrc::EYE_L);
		cv::UMat *eyeRightMat = currentframe->getImg(FrameSrc::EYE_R);
		cv::UMat *sceneMat = currentframe->getImg(FrameSrc::SCENE);

		//OpenCV only provides writing for Mat's
		eyeLeftWriter->write(eyeLeftMat->getMat(cv::ACCESS_READ));
		eyeRightWriter->write(eyeRightMat->getMat(cv::ACCESS_READ));
		sceneWriter->write(sceneMat->getMat(cv::ACCESS_READ));

		frameUpdated = false;
	}
}

