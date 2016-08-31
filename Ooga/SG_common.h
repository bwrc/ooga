#ifndef SG_COMMON_H
#define SG_COMMON_H

#include <chrono>

typedef std::chrono::high_resolution_clock hrclock;
typedef std::chrono::milliseconds msecs;

struct TGrabStatistics{
	msecs grabTime;
	msecs frameTime;
	msecs waitTime;
};

struct TTrackingResult{
	//this thingie should hold them results
	int val;
};


#endif