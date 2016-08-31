//#pragma once
#ifndef PERFORMANCETIMER
#define PERFORMANCETIMER

#include <chrono>
#include <vector>
#include <iostream>

typedef std::pair<double, std::string> timestampvalue;

class TPerformanceTimer
{
public:
	TPerformanceTimer();
	~TPerformanceTimer();

	void start();
	double elapsed_ms();

	void addTimeStamp(std::string msg);
	//std::vector<std::pair<double, std::string>> getTimeStamps();
	std::vector<timestampvalue> getTimeStamps();
	void dumpTimeStamps(std::ostream& dumpto, bool clear = true );
	void clearTimeStamps();

private:

	//std::chrono::steady_clock myclock;
	std::chrono::system_clock::time_point starttime;
	//std::vector<std::pair<double, std::string>> timeStamps;
	std::vector<timestampvalue> timeStamps;
};

#endif
