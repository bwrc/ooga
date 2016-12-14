#ifndef HELPERS_H
#define HELPERS_H

#include "OOGUI_OGL.h"

std::string readFile(const char *filePath) {
	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);

	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}

GLuint matToTexture(bool generateNewTexture, GLuint textureID, cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
	// Generate a number for our textureID's unique handle
	// unless we already have one
	if (generateNewTexture) {
		//GLuint textureID;
		glGenTextures(1, &textureID);
	}

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Set incoming texture format
	GLenum inputColourFormat = GL_BGR_EXT;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGB,            // Internal colour format to convert to
		mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
		mat.rows,          // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		mat.ptr());        // The actual image data itself

	glGenerateMipmap(GL_TEXTURE_2D);

	return textureID;
}

void makeQuad(int w, int h,
	std::vector<std::array<float, 3>> &quadpoints,
	std::vector<std::array<int, 3>> &faces,
	std::vector<std::array<float, 2>> &texCoords){

	GLfloat x, y;
	x = GLfloat(w);
	y = GLfloat(h);

	//quadpoints.push_back(std::array<float, 3>{{0.0f, 0.0f, 0.0f}});
	//quadpoints.push_back(std::array<float, 3>{{float(w), 0.0f, 0.0f}});
	//quadpoints.push_back(std::array<float, 3>{{float(w), float(h), 0.0f}});
	//quadpoints.push_back(std::array<float, 3>{{0.0f, float(h), 0.0f}});

	quadpoints.push_back(std::array<float, 3>{{-x/2, -y/2, 0.0f}});
	quadpoints.push_back(std::array<float, 3>{{x/2, -y/2, 0.0f}});
	quadpoints.push_back(std::array<float, 3>{{-x/2, y/2, 0.0f}});
	quadpoints.push_back(std::array<float, 3>{{x/2, y/2, 0.0f}});

	faces.push_back(std::array<int, 3>{ {0, 2, 1}});
	faces.push_back(std::array<int, 3>{ {1, 2, 3}});

	texCoords.push_back(std::array<float, 2>{{0.0f, 0.0f}});
	texCoords.push_back(std::array<float, 2>{{1.0f, 0.0f}});
	texCoords.push_back(std::array<float, 2>{{0.0f, 1.0f}});
	texCoords.push_back(std::array<float, 2>{{1.0f, 1.0f}});
}

// HELPERS END /////////////////////////////////////////////////////

#endif