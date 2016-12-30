OTHER_FILES += \
    cali.pro.user

HEADERS += \
    widget.h \
    pages.h \
    singlecam.h \
    calicam.h \
    twocam.h \
    mymath.h \
    calidatamethod.h \
    autograb.h \
    autograbtwocam.h

SOURCES += \
    widget.cpp \
    pages.cpp \
    main.cpp \
    singlecam.cpp \
    calicam.cpp \
    twocam.cpp \
    mymath.cpp \
    calidatamethod.cpp \
    autograb.cpp \
    autograbtwocam.cpp

RESOURCES += \
    application_resource.qrc


#INCLUDEPATH += /home/opencv-2.4.0/build/debug/include/
#LIBS += -L/home/opencv-2.4.0/build/debug/lib/ -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d

#INCLUDEPATH += /home/opencv-2.4.0/build/release/include/
#LIBS += -L/home/opencv-2.4.0/build/release/lib/ -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d

INCLUDEPATH += /home/miika/software/opencv-2.4.11/include/
LIBS += -L/home/miika/software/opencv-2.4.11/lib/ -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d

#QT += widgets  (for QT 5+ (will lead to other problems though)
