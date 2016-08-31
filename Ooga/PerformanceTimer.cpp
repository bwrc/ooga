#include "PerformanceTimer.h"

TPerformanceTimer::TPerformanceTimer()
{
	this->start();
}

TPerformanceTimer::~TPerformanceTimer()
{
}

void TPerformanceTimer::start()
{
	starttime = std::chrono::system_clock::now();
}

double TPerformanceTimer::elapsed_ms()
{
	//return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - starttime).count();
	return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now() - starttime).count();
}

void TPerformanceTimer::addTimeStamp(std::string msg)
{
	//timeStamps.push_back(std::pair<double, std::string>(this->elapsed_ms(), msg));
	timeStamps.push_back(timestampvalue(this->elapsed_ms(), msg));
}

//std::vector<std::pair<double, std::string>> TPerformanceTimer::getTimeStamps()
std::vector<timestampvalue> TPerformanceTimer::getTimeStamps()
{
	return timeStamps;
}

void TPerformanceTimer::dumpTimeStamps(std::ostream& dumpto, bool clear )
{
	for (auto& item : timeStamps) {
		dumpto << std::get<0>(item)
		<< " - "
		<< std::get<1>(item).c_str()
		<< std::endl;
	}
	if( clear )	timeStamps.clear();

}

void TPerformanceTimer::clearTimeStamps()
{
	timeStamps.clear();
}

