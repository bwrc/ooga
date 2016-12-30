#include "ResultHandler.h"
#include "boost/filesystem.hpp" //OS independent existence checking

namespace fs = boost::filesystem;

ResultHandler::ResultHandler()
{

}

ResultHandler::~ResultHandler()
{

}

bool ResultHandler::SetFile( std::string fn )
{
//    output = new std::ofstream(fn, std::ofstream::out);
//    output << boost::format("HEADER %1 %2 %3") % 1.0 % 2.0 % 3.0 << std::endl;
    bool fileOK = false;

    if( fs::is_directory(fn) ){
        std::cout << fn << " is a directory, please configure a proper output file." << std::endl;
        return false;
    }

    if( fs::exists(fn) && fs::is_regular_file(fn) ){
        char c;
        std::cout << "Configured result file " << fn << " already exists." << std::endl;
        std::cout << "(o)verwrite, or (q)uit?" << std::endl;
        std::cin >> c;
        switch(c){
            case 'o':
                std::cout << "Overwriting " << fn << std::endl;
                fileOK = true;
                break;
            default: //handles q as well
                std::cout << "Not allowed to overwrite " << fn << ". Shutting down." << std::endl;
                fileOK = false;
        }
    } else { //file doesn't exist, go on
        fileOK = true;
    }

    bool fileWriteable = false;
    if( fileOK){ 
        if(output=fopen( fn.c_str(), "w") ){ 
            fileWriteable = true;
        } else {
            fileWriteable = false;
            std::cout << "Could not access file " << fn << ". Shutting down" << std::endl;
        }
    }

    if( fileWriteable){
        return true;
    } else {
        return false;
    }
}

void ResultHandler::pushSample( TGazeTrackingResult sample )
{
/*    std::string str;
    output << boost::format("test\t%1%\t%2%\t%3%")
      % sample.timestamp
      % sample.pog.x
      % sample.pog.y
      << std::endl;
      */
      std::string state;
      switch(sample.state){
          case 0: state="BLINK"; break;
          case 1: state="RIGHT"; break;
          case 2: state="LEFT"; break;
          case 3: state="BOTH"; break;
      }

      fprintf(output, "%lu\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n", 
      sample.timestamp.count(),
      sample.pog.x,
      sample.pog.y,
      sample.gazedist,
      sample.score_l,
      sample.score_r,
      state.c_str());

}

void ResultHandler::writeHeader( std::string configFilename, hrclock::time_point zerotime)
{
    fprintf(output, "Using config filename: " );
    fprintf(output, configFilename.c_str());
    fprintf(output, "\nStart time %ud\n\n", std::chrono::time_point_cast<std::chrono::milliseconds>(zerotime));

    fprintf( output, "TIME\tPOG_X\tPOG_Y\tPOG_D\tSCORE_L\tSCORE_R\tSTATE\n");
}

bool ResultHandler::close()
{
    fclose(output);
}

