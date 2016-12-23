# compiler flags
CC=g++
#INC_DIRS= -I../ -I/usr/local/include/eigen3 -I/usr/local/include/
INC_DIRS= -I../ -I/usr/local/include/ -I/usr/local/include/eigen3 -Iboost_dir
#INCS=$(foreach d, $INC_DIR, -I$d)
CFLAGS = -c -std=c++11 $(INC_DIRS) -Wfatal-errors #-Wall
#LDFLAGS= -L/usr/local/lib/boost/ -L/usr/local/lib/
LDFLAGS= -Lboost_dir/stage/lib -L/usr/local/lib/ -L/usr/local/eigen3
SOURCES= Camera.cpp FrameRateLimiter.cpp \
VideoIOHandler.cpp Cornea_computer.cpp \
PerformanceTimer.cpp VideoWriterNoThread.cpp \
EyeTracker.cpp \
PupilEstimator.cpp FrameBinocular.cpp \
Settings.cpp FrameProcessor.cpp GlintFinder.cpp Ooga.cpp \
computePupilCenter3d.cpp getPupilCenter.cpp getPupilEllipsePoints.cpp kalmanFilterGazePoint.cpp \
../utils/OOGUI.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=ooga

#LIBRARIES
LIBS_opencv = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio
LIBS_boost = -lboost_thread -lboost_filesystem -lboost_program_options -lboost_system
LIBS_pthread = -lpthread
LIBS_GL = -lGL -lGLEW

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(LIBS_opencv) $(LIBS_boost) $(LIBS_pthread) $(LIBS_GL) $(OBJECTS) -o $@

pkgconf: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ `pkg-config opencv gsl glfw3 eigen3 --cflags --libs`  $(LIBS_pthread) $(LIBS_boost) $(LIBS_GL) 

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(EXECUTABLE)
