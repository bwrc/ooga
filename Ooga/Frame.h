//#pragma once
#ifndef FRAME
#define FRAME

/* TFrame : a class holding
- (pointers to) a pair of cv::Mat,
- unique timestamp and
- unique frame counter
*/

#include <opencv2/core/core.hpp>

// enumeration for identifying frame source
// if binocular, EYE -> EYE_L, EYE_R
enum class FrameSrc{ SCENE, EYE };

struct TTrackingResult{
	//this thingie should hold them results
	int val;
};

class TFrame
{
public:
	TFrame();
	~TFrame();

	void setImg(FrameSrc f, cv::UMat *img);
	cv::UMat *getImg(FrameSrc f);

	void setNumber(const int64 num){ number = num; };
	int64 getNumber(){ return number; };

	void setTimestamp(const int64 ts){ timestamp = ts; };
	int64 getTimestamp(){ return timestamp; };

	//for visualizing auxiliary processing results
	void pushAuxImg(cv::UMat *img);
	bool popAuxImg(cv::UMat *&img); //ref to pointer

	void setTrackingResult(TTrackingResult tr){ /*trackres = tr;*/ };
	TTrackingResult getTrackingResult(){ /*return trackres;*/ };

private:
	int64 timestamp;
	int64 number;
	cv::UMat *img_eye;
	cv::UMat *img_scene;

	TTrackingResult trackres;

	std::vector<cv::UMat *> aux_images;

};

#endif
