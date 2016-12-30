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
		std::cout << argv << std::endl;
		//get root path
		boost::filesystem::path ROOTPATH(boost::filesystem::initial_path<boost::filesystem::path>());
		ROOTPATH = ROOTPATH.parent_path();

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
#ifdef WIN32
			configFile = ROOTPATH.string() + "\\" + vm["config"].as< std::string >();
#else
			configFile = ROOTPATH.string() + "/" + vm["config"].as< std::string >();
#endif
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
			eyeVidLeftFile = ROOTPATH.string() + "/" + vm["eyefileL"].as< std::string >();
			//eyeVidLeftFile = rootpath + vm["eyefileL"].as< std::string >();

			//check if file exists here?
		}

		if (vm.count("eyefileR")){
			// command line <- --eyefile videos\eyeshot.mp4
			eyeVidRightFile = ROOTPATH.string() + "/" + vm["eyefileR"].as< std::string >();
			//eyeVidRightFile = rootpath + vm["eyefileR"].as< std::string >();

			//check if file exists here?
		}

		if (vm.count("scenefile")){
			sceneVidFile = ROOTPATH.string() + "/" + vm["scenefile"].as< std::string >();
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
		eyeLeftCam.type = 0; //camera
	}
	else {
		eyeLeftCam.type = 1; //video file
	}
	eyeLeftCam.feedNumber = tree.get<int>("settings.cameras.eyeleft.num", 0);
	eyeLeftCam.fileName = tree.get<std::string>("settings.cameras.eyeleft.file", "");
	eyeLeftCam.flip = tree.get<int>("settings.cameras.eyeleft.flip", 0);
// moved to calibration block	eyeLeftCam.calibrationFile = tree.get<std::string>("settings.cameras.eyeleft.cal", "eye.yaml");

	//RIGHT eye camera
	if (tree.get<std::string>("settings.cameras.eyeright.<xmlattr>.type") == "cam"){
		eyeRightCam.type = 0;
	}
	else {
		eyeRightCam.type = 1;
	}
	eyeRightCam.feedNumber = tree.get<int>("settings.cameras.eyeright.num", 0);
	eyeRightCam.fileName = tree.get<std::string>("settings.cameras.eyeright.file", "");
	eyeRightCam.flip = tree.get<int>("settings.cameras.eyeright.flip", 0);
	// moved to calibration block	eyeRightCam.calibrationFile = tree.get<std::string>("settings.cameras.eyeright.cal", "eye.yaml");

	//scene camera
	if (tree.get<std::string>("settings.cameras.scene.<xmlattr>.type") == "cam"){
		sceneCam.type = 0;
	}
	else {
		sceneCam.type = 1;
	}
	sceneCam.feedNumber = tree.get<int>("settings.cameras.scene.num", 0);
	sceneCam.fileName = tree.get<std::string>("settings.cameras.scene.file", "");
	sceneCam.flip = tree.get<int>("settings.cameras.scene.flip", 0);
	//sceneCam.calibrationFile = tree.get<std::string>("settings.cameras.scene.cal", "scene.yaml");

	if (tree.get<std::string>("settings.savefiles") == "true"){
		saveFrames = true;
	}
	else {
		saveFrames = false;
	}

	//calibration data
	camerarig_file = tree.get<std::string>("settings.calibration.camerarig.<xmlattr>.filename", "../calibration/camerarig.yaml");
	K9_file = tree.get<std::string>("settings.calibration.K9.<xmlattr>.filename", "../calibration/K9.yaml");
	CM_left_file = tree.get<std::string>("settings.calibration.CM_left.<xmlattr>.filename", "../calibration/file_CM_left");
	CM_right_file = tree.get<std::string>("settings.calibration.CM_right.<xmlattr>.filename", "../calibration/file_CM_right");
	glintmodel_file = tree.get<std::string>("settings.calibration.glintmodel.<xmlattr>.filename", "../calibration/glint_model.yaml");
	params_file = tree.get<std::string>("settings.calibration.parameters.<xmlattr>.filename", "../calibration/parameters.yaml");
	cam_left_eye_file = tree.get<std::string>("settings.calibration.cam_lefteye.<xmlattr>.filename", "../calibration/eye_cam_left.yaml");
	cam_right_eye_file = tree.get<std::string>("settings.calibration.cam_righteye.<xmlattr>.filename", "../calibration/eye_cam_right.yaml");
	LED_pos_file = tree.get<std::string>("settings.calibration.led_positions.<xmlattr>.filename", "../calibration/LED_positions.model.yaml");

	//result handling
	std::string tmp;
	video_folder="";
	result_file="";
	LSL_streamname="";
	tmp = tree.get<std::string>("settings.results.videos.<xmlattr>.save", "no");
	if(std::strcmp(tmp.c_str(), "yes")==0){
		saveVideos = true;
		video_folder = tree.get<std::string>("settings.results.videos.<xmlattr>.folder", "../videos/");
	}
	tmp = tree.get<std::string>("settings.results.resultfile.<xmlattr>.save", "no");
	if(std::strcmp(tmp.c_str(), "yes")==0){
		saveResults = true;
		result_file = tree.get<std::string>("settings.results.resultfile.<xmlattr>.filename", "../results/tmp.log");
	}
	tmp = tree.get<std::string>("settings.results.LSL.<xmlattr>.stream", "no");
	if(std::strcmp(tmp.c_str(), "yes")==0){
		streamLSL = true;
		LSL_streamname = tree.get<std::string>("settings.results.LSL.<xmlattr>.streamname", "OOGA_STREAM");
	}

}

void TSettings::saveSettings(std::string filename)
{

}
