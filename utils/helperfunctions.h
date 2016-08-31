
#include <fstream>								//for fileExists

//check if file exists / can be accessed (is not open by another process)
inline bool fileExists(const std::string& name) {
	std::ifstream f(name.c_str());
	if (f.good()) {
		f.close();
		return true;
	}
	else {
		f.close();
		return false;
	}
}
