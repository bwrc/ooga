//#pragma once
#ifndef FRAME_RATE_LIMITER_H
#define FRAME_RATE_LIMITER_H


#include <chrono>
#include <vector>
#include <numeric>

#define DEFAULT_BUFFER_SIZE 10

class FrameRateLimiter
{
public:
	FrameRateLimiter();
	~FrameRateLimiter();

	bool acquireSlot();
	virtual void notifyProcessed();
	virtual void notifyProcessed(/*std::chrono::duration<*/std::chrono::milliseconds/*>*/ timeProcessed);
	bool freeSlots(int n);
	void setBufferCapacity(int sz);
	int getBufferCapacity();
	double getAverageProcessingTime();

private:
	int slotsInUse;
	int usedSlotCounter;
	int divisor;
	int bufferCapacity;

	std::vector<std::chrono::milliseconds> durations;

};

#endif