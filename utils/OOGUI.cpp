//#include "stdafx.h"
#include "OOGUI.h"
#include "../utils/helpers.h"

OOGUI::OOGUI()
{
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(EXIT_FAILURE);
	}
	layout = 1;
}

OOGUI::~OOGUI()
{
	glDeleteTextures(1, &textureid[0]);
	glDeleteTextures(1, &textureid[1]);
	glDeleteTextures(1, &textureid[2]);
	glDeleteTextures(1, &textureid[3]);
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	glfwDestroyCursor(cursor_normal);
	glfwDestroyCursor(cursor_crosshair);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

void OOGUI::setSize(int w, int h) {
	width = w;
	height = h;
}

void OOGUI::SetLayout(int _layout){
	int w, h;
	float aspect = 1.0f;
	glfwGetWindowSize(window, &w, &h);
	//todo: check if w/h too big for screen
	switch (layout){
	case 1:
		aspect = 1280.0 / 960.0;
		break;
	case 2:
		aspect = 1.0;
		break;
	case 3:
		aspect = 1280.0 / 720.0;
		break;
	}
	glfwSetWindowSize(window, w, w / aspect);

	layout = _layout;
}

bool OOGUI::Initialize() {

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	//	glfwWindowHint(GLFW_SAMPLES, 4);

	// Open OpenGL window
	this->setSize(1280, 960);
	window = glfwCreateWindow(width, height, "OOGA :: version", NULL, NULL);

	//add this: install 3.2 glfwSetWindowAspectRatio(1280, 960);

	if (!window)
	{
		fprintf(stderr, "Failed to open GLFW window\n");

		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	//Make our window current
	//this was kinky, opening an OpenCV window with imshow
	//screwed up GLEW initialization (that context apparently became active)
	glfwMakeContextCurrent(window);

	//init glew - ONLY AFTER WINDOW CREATION!
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW " << err << std::endl;
		return false;
	}

	LoadShaders();

	// Set callback functions
	// needed for glfwGetUserPointer to work, this allows static members to be used as callbacks
	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
	glfwSetWindowRefreshCallback(window, windowRefreshCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetKeyCallback(window, keyCallback);

	// Enable vsync
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (glfwExtensionSupported("GL_ARB_multisample") ||
		glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR) >= 2 ||
		glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR) >= 3)
	{
		glEnable(GL_MULTISAMPLE_ARB);
	}

	glfwGetFramebufferSize(window, &width, &height);
	framebufferSizeFun(width, height);

	//this is the quad for displaying the video feeds on
	makeQuad(2,2, quadpoints, faces, texCoords); //ortho projection -> [-1,1] => 2

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	//vert.sha: in vec3 position
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadpoints), &quadpoints.front(), GL_STATIC_DRAW);
	const GLuint vertLoc(glGetAttribLocation(shaderProgram, "position"));
	glVertexAttribPointer(vertLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(vertLoc);

	//vert.sha: in vec2 texCoord
	glGenBuffers(1, &VBO_Tex);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Tex);
	glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float) * 2, &texCoords.front(), GL_STATIC_DRAW);
	texLoc = (glGetAttribLocation(shaderProgram, "texCoord"));
	glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(texLoc);

	//element buffer
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(int) * 3, &faces.front(), GL_STATIC_DRAW);

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	loadedAll = true;

	//create cursors
	cursor_normal = glfwCreateStandardCursor(GLFW_CURSOR_NORMAL);
	cursor_crosshair = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);

	return true;
}

bool OOGUI::LoadShaders() {
	std::string svertex = readFile("../utils/shaders/vert.sha");
	std::string sfrag = readFile("../utils/shaders/frag.sha");

	// Build and compile our shader program
	// Vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	const char *vertSrc = svertex.c_str();
	glShaderSource(vertexShader, 1, &vertSrc, NULL);
	glCompileShader(vertexShader);
	// Check for compile time errors
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	const char *fragSrc = sfrag.c_str();
	glShaderSource(fragmentShader, 1, &fragSrc, NULL);
	glCompileShader(fragmentShader);
	// Check for compile time errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Link shaders
	//moved to class GLuint
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	// Check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	if (!success) return false;
	return true;
}

auto OOGUI::framebufferSizeFun(int w, int h) -> void
{
	width = w;
	height = h > 0 ? h : 1;
	do_redraw = true;
}


auto OOGUI::windowRefreshFun(void) -> void
{
	drawAllViews();
	//glfwSwapBuffers(window);
	do_redraw = false;
}

void OOGUI::drawAllViews()
{
	if (!loadedAll) return;

	float aspect;

	// Calculate aspect of window
	if (height > 0)
		aspect = (float)width / (float)height;
	else
		aspect = 1.f;

	// Enable scissor test
	glEnable(GL_SCISSOR_TEST);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	int h_limit = 0;
	int w_limit = 0;

	switch (layout){
	case 1: //default 4x4
		// Upper left view (TOP VIEW)
		drawViewPort(1, 0, height / 2, width / 2, height / 2);
		//Upper right
		drawViewPort(2, width / 2, height / 2, width / 2, height / 2);
		// Lower left
		drawViewPort(3, 0, 0, width / 2, height / 2);
		//Lower right
		drawViewPort(4, width / 2, 0, width / 2, height / 2);
		break;
	case 2: //eyes on top
		h_limit = 3.0 / 4.0 * height;
		// left eye
		drawViewPort(1, 0, h_limit, width/3, height/4);
		//right eye
		drawViewPort(2, width / 3, h_limit, width / 3, height / 4);
		// scene
		drawViewPort(3, 0, 0, width, h_limit);
		//stats
		drawViewPort(4, width *2/3, h_limit, width / 3, height / 4);
		break;
	case 3: //eyes on right
		w_limit = 3.0 / 4.0 * width;
		// left eye
		drawViewPort(1, w_limit, height*2/3, width/4, height/3);
		//right eye
		drawViewPort(2, w_limit, height/3, width/4, height / 3);
		// scene
		drawViewPort(3, 0, 0, w_limit, height);
		//stats
		drawViewPort(4, w_limit, 0, width / 4, height / 3);
		break;
	}

	// Disable depth test
	glDisable(GL_DEPTH_TEST);

	// Disable scissor test
	glDisable(GL_SCISSOR_TEST);

	glfwSwapBuffers(window);
}

void OOGUI::drawViewPort(int num, int x, int y, int width, int height) {

	glViewport(x, y, width, height);
	glScissor(x, y, width, height);

	//clear viewport
						//++counter;
						//glClearColor(((counter % 1000) / 1000.0f), num/4.0f, num/4.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::mat4x4 view = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 1.0f),	//eye
		glm::vec3(0.0f, 0.0f, 0.0f),    //center
		glm::vec3(0.0f, 1.0f, 0.0f));   //up

	glm::mat4x4 model = glm::mat4(); //identity

	glm::mat4x4 projection =
			glm::ortho(0.0f, float(width), 0.0f, float(height), 0.0f, 5.0f);
			//glm::perspective(45.0f, (GLfloat)640 / (GLfloat)480, 0.1f, 100.0f);

	// Get the uniform locations
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
	GLint projLoc = glGetUniformLocation(shaderProgram, "projection");

	// Pass the matrices to the shader
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));//

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, quadpoints.size()*sizeof(float) * 3, &quadpoints.front(), GL_DYNAMIC_DRAW);

	glUseProgram(shaderProgram);

	glUniform1i(texLoc, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureid[num-1]);

	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 6, GL_FLOAT, GL_FALSE, 3, (void*)0);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, faces.size() * 3, GL_UNSIGNED_INT, 0);

	//unbind
	glBindVertexArray(0);
	glUseProgram(0);

	//glfwSwapBuffers(window);
}

//========================================================================
// Mouse position callback function
//========================================================================
void OOGUI::cursorPosFun(double x, double y)
{
	int wnd_width, wnd_height, fb_width, fb_height;
	double scale;

	glfwGetWindowSize(window, &wnd_width, &wnd_height);
	glfwGetFramebufferSize(window, &fb_width, &fb_height);

	scale = (double)fb_width / (double)wnd_width;

	x *= scale;
	y *= scale;

	//scale mouse clicks to correspond to original video coordinates
	int hw = width / 2;
	int hh = height / 2;
	int currentlyOnTopOf = 0;

	if (xpos >= hw)
		currentlyOnTopOf+= 1;
	if (ypos >= hh)
		currentlyOnTopOf += 2;

	int sx, sy; //scaled

	sx = fmod(x, hw) / float(hw) * 640;
	sy = fmod(y, hh) / float(hh) * 480;

//	std::cerr << "cursor at " << sx << ", " << sy << "in win " << currentlyOnTopOf << std::endl;

/*	switch (currentlyOnTopOf)
	{
	case 1:
		do_redraw = true;
		break;
	case 3: //scene
		do_redraw = true;
		break;
	case 4:
		do_redraw = true;
		break;
	default:
		break;
	}
*/
	// Remember cursor position
	xpos = x;
	ypos = y;
}

//========================================================================
// Mouse button callback function
//========================================================================
void OOGUI::mouseButtonFun( int button, int action, int mods)
{
	if ((button == GLFW_MOUSE_BUTTON_LEFT) && action == GLFW_PRESS)
	{
		// Detect which of the four views was clicked
		//here coords from upper left
		switch (layout){
		case 1: //2x2
			active_view = 1;
			if (xpos >= width / 2)
				active_view += 1;
			if (ypos >= height / 2)
				active_view += 2;
			break;
		case 2: //eyes on top
			active_view = 1;
			if (ypos >= height / 4.0){ active_view = 3; }
			else {
				if (xpos >= 2.0 / 3.0 * width){ active_view = 4; }
				else if (xpos >= 1.0 / 3.0*width){ active_view = 2; }
			}
			break;
		case 3: //eyes on right
			active_view = 3;
			if (xpos >= 3.0 / 4.0 * width){
				if (ypos >= 2.0 / 3.0*height){ active_view = 4; }
				else if (ypos >= 1.0 / 3.0*height){ active_view = 2; }
				else{ active_view = 1; }
			}
			break;
		}

		//if click on scene and calibration samples are collected
		if ((active_view == 3) && getCalibrationSamples){

			int wnd_width, wnd_height, fb_width, fb_height;

			glfwGetWindowSize(window, &wnd_width, &wnd_height);
			glfwGetFramebufferSize(window, &fb_width, &fb_height);

			double scale = (double)fb_width / (double)wnd_width;
			double x = xpos * scale;
			double y = ypos * scale;

			double hw, hh;
			double offset_h, offset_w;

			//scale mouse clicks to correspond to original video coordinates
			switch (layout){
			case 1: //2x2, half of height
				hw = width / 2;
				hh = height / 2;
				offset_w = 0;
				offset_h = height / 2;
				break;
			case 2: //eyes on top
				hw = width;
				hh = 3.0 / 4.0*height;
				offset_w = 0;
				offset_h = height / 4.0;
				break;
			case 3:
				hw = 3.0 / 4.0*width;
				hh = height;
				offset_w = 0;
				offset_h = 0;
			}

			double sx, sy; //scaled

			sx = fmod((x - offset_w), hw) / hw*640.0;
			sy = fmod((y - offset_h), hh) / hh*640.0;

			//		sx = fmod(x, hw) / float(hw) * 640;
			//		sy = fmod(y, hh) / float(hh) * 480;
			calCallback(sx, sy);
			//std::cout << "add calib sample: " << sx << ", " << sy << std::endl;

			//todo: addCalibrationSample(sx, sy);
		}
	}
	do_redraw = true;
}

void OOGUI::SetCallBackFunction(std::function<void(RunningModes mode, bool value)> callback){
	modeCallBack = callback;
}

void OOGUI::SetCalibrationCallback( std::function<void(double x, double y)> cal_callback){
	calCallback = cal_callback;
}


void OOGUI::key_callback(int key, int scancode, int action, int mods)
{
	//std::cout << "key: " << key << std::endl;
	if (action == GLFW_PRESS){ //only consider first presses
		switch (key){
		case GLFW_KEY_ESCAPE:
			//TODO: killwindowcallback
			modeCallBack(OOGA_MODE_RUNNING, false);
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_C:
			getCalibrationSamples = !getCalibrationSamples; // toggle calibration on/off
			if (getCalibrationSamples){
				glfwSetCursor(window, cursor_crosshair);
			}
			else {
				glfwSetCursor(window, cursor_normal);
			}
			modeCallBack(OOGA_MODE_CALIBRATE, getCalibrationSamples);
			break;
		case GLFW_KEY_N:
			//todo: oogaCallBack( OOGA_NEXT_FRAME );
			break;
		case GLFW_KEY_R:
			//todo: oogaCallBack( OOGA_MODE_CALIBRATE, true );
			break;
		case GLFW_KEY_SPACE:
			//todo: oogaCallBack( OOGA_PAUSE );
			break;
		case GLFW_KEY_1:
			SetLayout(1);
			break;
		case GLFW_KEY_2:
			SetLayout(2);
			break;
		case GLFW_KEY_3:
			SetLayout(3);
			break;
		default:
			std::cout << "unknown key: " << key << std::endl;
		}
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

bool OOGUI::update() {
	// Only redraw if we need to
	if (do_redraw)
		windowRefreshFun();

	// Check for new events
	glfwPollEvents();

	// Check if the window should be closed
	if (glfwWindowShouldClose(window))
		return false;

	return true;
}

void OOGUI::pushFrame(TBinocularFrame &frame) {
	if (firstrun){ // create textures if first round, otherwise just update
		firstrun = false;
		textureid[0] = matToTexture(true, 0, (frame.getImg(FrameSrc::EYE_R))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		textureid[1] = matToTexture(true, 0, (frame.getImg(FrameSrc::EYE_L))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		textureid[2] = matToTexture(true, 0, (frame.getImg(FrameSrc::SCENE))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		//generate an empty mat for the stats window as the frame only has three
		textureid[3] = matToTexture(true, 0, cv::Mat(cv::Size(640, 480), CV_8UC3), GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}
	else {
		matToTexture(false, textureid[0], (frame.getImg(FrameSrc::EYE_R))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		matToTexture(false, textureid[1], (frame.getImg(FrameSrc::EYE_L))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		matToTexture(false, textureid[2], (frame.getImg(FrameSrc::SCENE))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		//matToTexture(false, textureid[3], (frame.getImg(FrameSrc::SCENE))->getMat(cv::ACCESS_READ), GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}
	do_redraw = true;
}


void OOGUI::pushFrame(std::vector<cv::Mat> &frame) {

	if (firstrun) { // create textures if first round, otherwise just update
		firstrun = false;
		textureid[0] = matToTexture(true, 0, frame.at(0), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		textureid[1] = matToTexture(true, 0, frame.at(1), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		textureid[2] = matToTexture(true, 0, frame.at(2), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		textureid[3] = matToTexture(true, 0, frame.at(3), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		//textureid = matToTexture(true, 0, frame, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}
	else {

		matToTexture(false, textureid[0], frame.at(0), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		matToTexture(false, textureid[1], frame.at(1), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		matToTexture(false, textureid[2], frame.at(2), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		matToTexture(false, textureid[3], frame.at(3), GL_NEAREST, GL_NEAREST, GL_CLAMP);
		//matToTexture(false, 0, frame, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}

	do_redraw = true;
}

void OOGUI::shutDown() {
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}
