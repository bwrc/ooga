//#pragma once
#ifndef SETTINGS_H
#define SETTINGS_H

/* Settings.h
Storing and restoring settings data. Makes use of boost::property_tree
*/

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/xml_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include <string>

#ifdef WIN32
#include <tchar.h>
#endif

#define DEFAULT_CONFIG_FILE "config.xml"

struct feedSource
{
	int type; // 0 = cam, 1 = file, 2 = realsense?
	int feedNumber; //USB cam enumeration
	std::string calibrationFile;
	std::string fileName;
	int flip;
};

class TSettings
{
public:
	TSettings();
	~TSettings();

#ifdef _WIN32
	bool processCommandLine(int argc, _TCHAR* argv[]);
#else
	bool processCommandLine(int argc, char** argv);
#endif

	void loadSettings(const std::string filename);
	void saveSettings(const std::string filename);

	// keep settings params here in public for easier access
	// bad design :)
	std::string configFile;
	std::string eyeVidLeftFile;
	std::string eyeVidRightFile;
	std::string sceneVidFile;

	feedSource eyeLeftCam;
	feedSource eyeRightCam;
	feedSource sceneCam;

	std::string camerarig_file; //this includes the scene calibration -> should move to cam_scene_file?
	std::string K9_file;
	std::string CM_left_file;
	std::string CM_right_file;
	std::string glintmodel_file;
	std::string params_file;
	std::string LED_pos_file;

	std::string cam_left_eye_file;
	std::string cam_right_eye_file;

	int nOfCams = 3;

	bool saveFrames = false;

};

#endif //SETTINGS_H
