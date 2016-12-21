#ifndef OOGUI_H
#define OOGUI_H

//#define GLEW_STATIC
#include <GL/glew.h>
#define GLFW_INCLUDE_GLU
#define GLFW_INCLUDE GLEXT
#include "glfw/glfw3.h"

#include <glm/glm.hpp> //GL Math
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <array>
#include <functional>
//#include <TBinocularFrame>

//loading shaders from file
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "../Ooga/FrameBinocular.h"
#include "../Ooga/oogaConstants.h"

class OOGUI
{
public:
	OOGUI();
	~OOGUI();

	bool Initialize();
	void pushFrame(std::vector<cv::Mat> &frame);
	void pushFrame(TBinocularFrame &frame);
	//void pushFrame(cv::Mat frame);
	void shutDown();
	void setSize(int w, int h);
	bool update();
	void drawAllViews();

	void drawViewPort(int num, int x, int y, int width, int height);
	void SetCallBackFunction(std::function<void(RunningModes mode, bool value)> callback);

	void RenderState(int viewport);


private:

	int counter = 0;

	GLFWwindow* window;

	GLuint VBO, VAO, EBO, VBO_Tex;
	GLuint texLoc;
	GLuint textureid[4];

	GLuint shaderProgram;
	GLuint fragmentShader;
	GLuint vertexShader;

	std::function<void(RunningModes, bool)> modeCallBack;

	std::vector<std::array<float, 3>> quadpoints;
	std::vector<std::array<int, 3>> faces;
	std::vector<std::array<float, 2>> texCoords;

	bool firstrun = true;

	bool LoadShaders();

	//window callbacks
	//see http://gamedev.stackexchange.com/questions/58541/how-can-i-associate-a-key-callback-with-a-wrapper-class-instance
	// static functions only work in the case of a single GLFW window
	inline static auto framebufferSizeCallback(GLFWwindow* window, int w, int h)
		-> void {
		OOGUI* win = static_cast<OOGUI*>( glfwGetWindowUserPointer(window));
		win->framebufferSizeFun(w, h);
	}
	inline static auto windowRefreshCallback(GLFWwindow* window)
		-> void {
		OOGUI* win = static_cast<OOGUI*>(glfwGetWindowUserPointer(window));
		win->windowRefreshFun();
	}
	inline static auto cursorPosCallback(GLFWwindow* window, double x, double y)
		-> void {
		OOGUI* win = static_cast<OOGUI*>(glfwGetWindowUserPointer(window));
		win->cursorPosFun(x, y);
	}
	inline static auto mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
		-> void {
		OOGUI* win = static_cast<OOGUI*>(glfwGetWindowUserPointer(window));
		win->mouseButtonFun(button, action, mods);
	}
	inline static auto keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
		-> void {
		OOGUI* win = static_cast<OOGUI*>(glfwGetWindowUserPointer(window));
		win->key_callback(key, scancode, action, mods);
	}

	auto framebufferSizeFun( int w, int h) -> void;
	auto windowRefreshFun(void) -> void;
	auto cursorPosFun(double x, double y) -> void;
	auto mouseButtonFun(int button, int action, int mods) -> void;
	auto key_callback(int key, int scancode, int action, int mods) -> void;

	// Mouse position
	double xpos = 0, ypos = 0;
	// Window size
	int width, height;
	// Active view: 0 = none, 1 = upper left, 2 = upper right, 3 = lower left,
	// 4 = lower right
	int active_view = 0;
	// Do redraw?
	bool do_redraw = 1;
	bool redraws[4]; //for each window part

	bool loadedAll = false;

	bool getCalibrationSamples = false;

	GLFWcursor *cursor_normal, *cursor_crosshair;

};


/*class OOGUI_OGL
{
public:
	OOGUI_OGL();
	~OOGUI_OGL();

	bool Initialize();
	//void pushFrame(std::vector<cv::Mat> frame);
	void pushFrame(cv::Mat frame);
	void shutDown();
	void setSize(int w, int h);
	bool update();
	void drawAllViews();

	void drawViewPort(int num, int x, int y, int width, int height);

private:

	int counter = 0;

	GLFWwindow* window;
	//GLuint shaderProgram;

	//GLuint VBO, VAO, EBO, VBO_Tex;
	//GLuint texLoc;
	GLuint textureid;

	std::vector<std::array<float, 3>> quadpoints;
	std::vector<std::array<int, 3>> faces;
	std::vector<std::array<float, 2>> texCoords;

	bool firstrun = true;

	//bool LoadShaders();

	//window callbacks
	//see http://gamedev.stackexchange.com/questions/58541/how-can-i-associate-a-key-callback-with-a-wrapper-class-instance
	// static functions only work in the case of a single GLFW window
	inline static auto framebufferSizeCallback(GLFWwindow* window, int w, int h)
		-> void {
		OOGUI_OGL* win = static_cast<OOGUI_OGL*>(glfwGetWindowUserPointer(window));
		win->framebufferSizeFun(w, h);
	}
	inline static auto windowRefreshCallback(GLFWwindow* window)
		-> void {
		OOGUI_OGL* win = static_cast<OOGUI_OGL*>(glfwGetWindowUserPointer(window));
		win->windowRefreshFun();
	}
	inline static auto cursorPosCallback(GLFWwindow* window, double x, double y)
		-> void {
		OOGUI_OGL* win = static_cast<OOGUI_OGL*>(glfwGetWindowUserPointer(window));
		win->cursorPosFun(x, y);
	}
	inline static auto mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
		-> void {
		OOGUI_OGL* win = static_cast<OOGUI_OGL*>(glfwGetWindowUserPointer(window));
		win->mouseButtonFun(button, action, mods);
	}
	inline static auto keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
		-> void {
		OOGUI_OGL* win = static_cast<OOGUI_OGL*>(glfwGetWindowUserPointer(window));
		win->key_callback(key, scancode, action, mods);
	}

	auto framebufferSizeFun(int w, int h) -> void;
	auto windowRefreshFun(void) -> void;
	auto cursorPosFun(double x, double y) -> void;
	auto mouseButtonFun(int button, int action, int mods) -> void;
	auto key_callback(int key, int scancode, int action, int mods) -> void;

	// Mouse position
	double xpos = 0, ypos = 0;
	// Window size
	int width, height;
	// Active view: 0 = none, 1 = upper left, 2 = upper right, 3 = lower left,
	// 4 = lower right
	int active_view = 0;
	// Do redraw?
	bool do_redraw = 1;

	bool loadedAll = false;

};
*/

#endif