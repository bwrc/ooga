#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H

#include <queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <Eigen/Eigen>

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

	//void push(Data const& data)
	void push(Data data)
//	void push(std::shared_ptr<Data> data)
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

#endif
