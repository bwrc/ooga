# compiler flags
CC=g++
#INC_DIRS= -I../ -I/usr/local/include/eigen3 -I/usr/local/include/
INC_DIRS= -I../ -I/usr/local/include/ -Ieigen_dir -Iboost_dir
#INCS=$(foreach d, $INC_DIR, -I$d)
CFLAGS = -c -Wall -std=c++11 $(INC_DIRS) -Wfatal-errors
#LDFLAGS= -L/usr/local/lib/boost/ -L/usr/local/lib/
LDFLAGS= -Lboost_dir/stage/lib -L/usr/local/lib/
SOURCES= Camera.cpp FrameRateLimiter.cpp \
orderGlints.cpp VideoIOHandler.cpp Cornea_computer.cpp \
PerformanceTimer.cpp VideoWriterNoThread.cpp \
EyeTracker.cpp \
PupilEstimator.cpp FrameBinocular.cpp \
Settings.cpp FrameProcessor.cpp GlintFinder.cpp Ooga.cpp \
computePupilCenter3d.cpp getPupilCenter.cpp getPupilEllipse.cpp getPupilEllipsePoints.cpp

#getGlints.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=ooga

#LIBRARIES
LIBS_opencv = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio
LIBS_boost = -lboost_thread -lboost_filesystem -lboost_program_options -lboost_system
LIBS_pthread = -lpthread

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(LIBS_opencv) $(LIBS_boost) $(LIBS_pthread) $(OBJECTS) -o $@

pkgconf: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ `pkg-config opencv gsl --cflags --libs`  $(LIBS_pthread) $(LIBS_boost)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(EXECUTABLE)
