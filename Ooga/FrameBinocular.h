//#pragma once
#ifndef FRAMEBINOCULAR
#define FRAMEBINOCULAR

/* TFrame : a class holding
- (pointers to) a pair of cv::Mat,
- unique timestamp and
- unique frame counter
*/

#include <opencv2/core/core.hpp>
#include <iostream>
#include "SG_common.h"

// enumeration for identifying frame source
// if binocular, EYE -> EYE_L, EYE_R
//enum class FrameSrc{ SCENE, EYE_L, EYE_R };
enum class FrameSrc{ EYE_R, EYE_L, SCENE };

class TBinocularFrame
{
public:
	TBinocularFrame();
	~TBinocularFrame();

	//TBinocularFrame & operator=(const TBinocularFrame& other);

	void setImg(FrameSrc f, cv::UMat *img);
	cv::UMat *getImg(FrameSrc f);

	void setNumber(const int64 num){ number = num; };
	int64 getNumber(){ return number; };

	void setTimestamp(const msecs ts){ timestamp = ts; };
	msecs getTimestamp(){ return timestamp; };

	//for visualizing auxiliary processing results
	void pushAuxImg(cv::UMat *img);
	bool popAuxImg(cv::UMat *&img); //ref to pointer

	void setGrabbingStats(TGrabStatistics gs){ this->grabStats = gs; }

	void setTrackingResult(TTrackingResult tr){ /*trackres = tr;*/ };
	TTrackingResult getTrackingResult(){ /*return trackres;*/ };

private:

	//by hiding these unique_ptr copy shouldn't be an issue on VS?
	//TBinocularFrame(const TBinocularFrame &obj);
	//TBinocularFrame(const TBinocularFrame &&obj);

	msecs timestamp;
	int64 number;
	cv::UMat *img_eye_L;
	cv::UMat *img_eye_R;
	cv::UMat *img_scene;

	TGrabStatistics grabStats;
	TTrackingResult trackres;

	std::vector<cv::UMat *> aux_images;

};

#endif
