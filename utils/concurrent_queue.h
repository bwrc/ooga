#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H

#include <queue>
#include <vector>
#include <numeric> //accumulate

//TODO: this still uses boost mutexes for condition variable
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

//#include <Eigen/Eigen>

#define DEFAULT_TOKENS 10
#define DEFAULT_REPORTED_TIMES 10

template<typename Data>

class concurrent_queue
{
private:
	std::queue<Data> myQueue;
	//std::queue<Data, Eigen::aligned_allocator<Data>> myQueue;
	mutable boost::mutex myMutex;
	boost::condition_variable myCondVar;

public:

	int getQueueLength()
	{
		// just reading, no need to mutex, right?
		return myQueue.size();
	}

	void clear()
	{
		try{
			boost::mutex::scoped_lock lock(myMutex);
			while (!myQueue.empty()){
				myQueue.pop();
			}
		}
		catch (int e){
			std::cout << "oh crap" << std::endl;
		}
	}

	void push(Data data)
	{
		boost::mutex::scoped_lock lock(myMutex);
		myQueue.push(data);
		lock.unlock();
		myCondVar.notify_one();
	}

	bool empty() const
	{
		boost::mutex::scoped_lock lock(myMutex);
		return myQueue.empty();
	}

	bool try_pop(Data& popped_value)
	{
		boost::mutex::scoped_lock lock(myMutex);
		if (myQueue.empty())
		{
			return false;
		}

		popped_value = std::move(myQueue.front());
		myQueue.pop();
		return true;
	}

	void wait_and_pop(Data& popped_value)
	{
		boost::mutex::scoped_lock lock(myMutex);
		while (myQueue.empty())
		{
			myCondVar.wait(lock);
		}

		popped_value = std::move(myQueue.front());
		myQueue.pop();
	}

};

template<typename Data>
class BalancingQueue : public concurrent_queue<Data>{
private:

	int tokens;
	int maxTokens;
	int reportedTimes;

	std::vector<double> consumerTimes;

	bool getToken();

	mutable boost::mutex balMutex;

public:
	BalancingQueue();
	~BalancingQueue();

	void setMaxTokens(int l);
	int getMaxTokens();
	bool try_push(Data data);
	void reportConsumerTime(double milliseconds);
	double getAverageConsumerTime();

	bool try_pop(Data& popped_value);
	void wait_and_pop(Data& popped_value);

};

template<typename Data>
BalancingQueue<Data>::BalancingQueue()
{
	this->tokens = 0;
	this->maxTokens = DEFAULT_TOKENS;
	consumerTimes.clear();
	reportedTimes = DEFAULT_REPORTED_TIMES;
}

template<typename Data>
BalancingQueue<Data>::~BalancingQueue()
{
}

template<typename Data>
bool BalancingQueue<Data>::getToken()
{
	boost::mutex::scoped_lock lock(balMutex);

	if (this->tokens < this->maxTokens){
		++tokens;
		return true;
	}
	else {
		return false;
	}
}

template<typename Data>
void BalancingQueue<Data>::setMaxTokens(int l)
{
	boost::mutex::scoped_lock lock(balMutex);

	if (l > 0) maxTokens = l;
}

template<typename Data>
int BalancingQueue<Data>::getMaxTokens()
{
	boost::mutex::scoped_lock lock(balMutex);

	return maxTokens;
}

template<typename Data>
bool BalancingQueue<Data>::try_push(Data data)
{
	if (this->getToken()){
		this->push(data);
		return true;
	}
	else {
		return false;
	}
}

template<typename Data>
void BalancingQueue<Data>::reportConsumerTime(double milliseconds)
{
	boost::mutex::scoped_lock lock(balMutex);

	if (consumerTimes.size()>reportedTimes){
		//pop first to keep length
		consumerTimes.erase(consumerTimes.begin());
	}
	consumerTimes.push_back(milliseconds);
}

template<typename Data>
double BalancingQueue<Data>::getAverageConsumerTime()
{
	boost::mutex::scoped_lock lock(balMutex);

	if (consumerTimes.size() > 0){
		double sum = 0.0;
		for (auto &x : consumerTimes){
			sum += x;
		}
		return sum / consumerTimes.size();
	}
	else {
		return 0.0;
	}
}

template<typename Data>
bool BalancingQueue<Data>::try_pop(Data& popped_value)
{

	if (concurrent_queue<Data>::try_pop(popped_value)){
		--tokens;
		return true;
	}
	else{
		return false;
	}
}

template<typename Data>
void BalancingQueue<Data>::wait_and_pop(Data& popped_value)
{
	concurrent_queue<Data>::wait_and_pop(popped_value);
	--tokens;

}


#endif
