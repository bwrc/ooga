#include "FrameRateLimiter.h"

FrameRateLimiter::FrameRateLimiter()
{
	slotsInUse = 0;
	usedSlotCounter = 0;
	bufferCapacity = DEFAULT_BUFFER_SIZE;
}


FrameRateLimiter::~FrameRateLimiter()
{

}


bool 
FrameRateLimiter::acquireSlot()
{
	if (slotsInUse < bufferCapacity) {
		++slotsInUse;
		++usedSlotCounter;
		return true;
	}
	else {
		return false;
	}
}

void
FrameRateLimiter::notifyProcessed() 
{
	if (slotsInUse > 0) --slotsInUse;
	
}

void 
FrameRateLimiter::notifyProcessed( std::chrono::milliseconds timeProcessed )
{
	//free a slot
	if( slotsInUse > 0) --slotsInUse;

	//update duration vector
	if (durations.size() < bufferCapacity) {
		durations.push_back(timeProcessed);
	}
	else {
		durations.at(usedSlotCounter%bufferCapacity) = timeProcessed;
	}
}

bool 
FrameRateLimiter::freeSlots(int n)
{
	if (bufferCapacity > n) {
		slotsInUse -= n;
		return true;
		//...and remove which durations, exactly?
	}
	return false;
}

void
FrameRateLimiter::setBufferCapacity( int sz)
{
	if(sz>0)
		bufferCapacity = sz;
	//insert memory volume test, etc.
	
}

int FrameRateLimiter::getBufferCapacity()
{
	return bufferCapacity;
}

double
FrameRateLimiter::getAverageProcessingTime() 
{
	std::chrono::milliseconds sum = std::chrono::milliseconds(0);
	for (std::chrono::milliseconds t : durations) {
		sum += t;
	}
	return sum.count() / durations.size();

}
