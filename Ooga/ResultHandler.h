#ifndef RESULT_HANDLER_H
#define RESULT_HANDLER_H

#include "SG_common.h"
#include <string>
#include <iostream>
#include <fstream>
//#include "boost/format.hpp"

class ResultHandler {
public:
    ResultHandler();
    ~ResultHandler();
    bool SetFile( std::string fn );
    //initialize?
    void pushSample( TGazeTrackingResult sample );
    void writeHeader( std::string configFilename, hrclock::time_point zerotime);
    bool close();

private:
    //std::ofstream* output;
    FILE* output;
};

#endif