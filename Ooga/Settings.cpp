#include "Settings.h"
#include <iostream>

namespace po = boost::program_options;
namespace pt = boost::property_tree;

TSettings::TSettings()
{
}

TSettings::~TSettings()
{
}

#ifdef _WIN32
bool TSettings::processCommandLine(int argc, _TCHAR* argv[])
#else
bool TSettings::processCommandLine(int argc, char** argv)
#endif
{
	try{

		//get root path
		boost::filesystem::path ROOTPATH(boost::filesystem::initial_path<boost::filesystem::path>());
		ROOTPATH = ROOTPATH.parent_path();

		//std::string rootpath = "c:/Code/SmarterTracker/";

		//define help text
		po::options_description poDesc("Allowed options");
		poDesc.add_options()
			("help", "show help")
			("eyefileL", po::value<std::string>(), "set eye cam file to use instead of camera")
			("eyefileR", po::value<std::string>(), "set eye cam file to use instead of camera")
			("scenefile", po::value<std::string>(), "set scene cam file to use instead of camera")
			("config", po::value<std::string>(), "set config file to read from.")
			;

		//parse command line
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, poDesc), vm);
		po::notify(vm);

		if (vm.count("help")){
			std::cout << poDesc << std::endl;
			return false;
		}

		if (vm.count("config")){
			configFile = ROOTPATH.string() + "\\" + vm["config"].as< std::string >();
			//configFile = rootpath + vm["config"].as< std::string >();
			std::cout << "Using config file " << configFile << std::endl;
		}
		else {
#ifdef _WIN32
			configFile = ROOTPATH.string() + "\\config\\" + std::string(DEFAULT_CONFIG_FILE);
			//configFile = rootpath + "\\config\\" + std::string(DEFAULT_CONFIG_FILE);
#else
			configFile = ROOTPATH.string() + "/config/" + std::string(DEFAULT_CONFIG_FILE);
#endif
			std::cout << "Using DEFAULT config file " << configFile << std::endl;
		}

		if (vm.count("eyefileL")){
			// command line <- --eyefile videos\eyeshot.mp4
			eyeVidLeftFile = ROOTPATH.string() + "\\" + vm["eyefileL"].as< std::string >();
			//eyeVidLeftFile = rootpath + vm["eyefileL"].as< std::string >();

			//check if file exists here?
		}

		if (vm.count("eyefileR")){
			// command line <- --eyefile videos\eyeshot.mp4
			eyeVidRightFile = ROOTPATH.string() + "\\" + vm["eyefileR"].as< std::string >();
			//eyeVidRightFile = rootpath + vm["eyefileR"].as< std::string >();

			//check if file exists here?
		}

		if (vm.count("scenefile")){
			sceneVidFile = ROOTPATH.string() + "\\" + vm["scenefile"].as< std::string >();
			//sceneVidFile = rootpath + "\\" + vm["scenefile"].as< std::string >();
			//if (fileExists(sceneVidFile)){
			//	std::cout << "Using " << sceneVidFile << " for scene video" << std::endl;
			//}
			//else {
			//	std::cout << "Invalid scenefile specified: " << sceneVidFile << std::endl;
			//	return 0;
			//}
		}
	}
	catch (int e){
		std::cout << "error processing settings: " << e << std::endl;
		return false;
	}

	return true;
}

void TSettings::loadSettings(std::string filename)
{
	pt::ptree tree;

	//parse the settings to a property tree
	pt::read_xml(filename, tree);

	//LEFT eye camera
	if (tree.get<std::string>("settings.cameras.eyeleft.<xmlattr>.type") == "cam"){
		eyeLeftCam.type = 0;
	}
	else {
		eyeLeftCam.type = 1;
	}
	eyeLeftCam.feedNumber = tree.get<int>("settings.cameras.eyeleft.num", 0);
	eyeLeftCam.fileName = tree.get<std::string>("settings.cameras.eyeleft.file", "");
	eyeLeftCam.calibrationFile = tree.get<std::string>("settings.cameras.eyeleft.cal", "eye.yaml");

	//RIGHT eye camera
	if (tree.get<std::string>("settings.cameras.eyeright.<xmlattr>.type") == "cam"){
		eyeRightCam.type = 0;
	}
	else {
		eyeRightCam.type = 1;
	}
	eyeRightCam.feedNumber = tree.get<int>("settings.cameras.eyeright.num", 0);
	eyeRightCam.fileName = tree.get<std::string>("settings.cameras.eyeright.file", "");
	eyeRightCam.calibrationFile = tree.get<std::string>("settings.cameras.eyeright.cal", "eye.yaml");

	//scene camera
	if (tree.get<std::string>("settings.cameras.scene.<xmlattr>.type") == "cam"){
		sceneCam.type = 0;
	}
	else {
		sceneCam.type = 1;
	}
	sceneCam.feedNumber = tree.get<int>("settings.cameras.scene.num", 0);
	sceneCam.fileName = tree.get<std::string>("settings.cameras.scene.file", "");
	sceneCam.calibrationFile = tree.get<std::string>("settings.cameras.scene.cal", "scene.yaml");

	if (tree.get<std::string>("settings.savefiles") == "true"){
		saveFrames = true;
	}
	else {
		saveFrames = false;
	}


}

void TSettings::saveSettings(std::string filename)
{

}
