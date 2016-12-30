#include "autograbtwocam.h"

AutoGrabTwoCam::AutoGrabTwoCam()
{
    ini();

}

void AutoGrabTwoCam::clearTimeStamps(){
    for (int i=0; i<N; i++) timeStamps[i]=0;

}


void AutoGrabTwoCam::ini(){

    clearTimeStamps();

    frameCount = 0;
    goodImagePairCount = 0;

    myTime.start();

}


void AutoGrabTwoCam::shift2left(
        bool eyePatternFoundFlag,
        bool scenePatternFoundFlag,
        double eyePointsCenterX,
        double scenePointsCenterX,
        double eyePointsCenterY,
        double scenePointsCenterY
        )
{
    for (int i=N-1; i>0; i--){
        timeStamps[i]=timeStamps[i-1];

        eyePointsCentersX[i]=eyePointsCentersX[i-1];
        scenePointsCentersX[i]=scenePointsCentersX[i-1];


        eyePointsCentersY[i]=eyePointsCentersY[i-1];
        scenePointsCentersY[i]=scenePointsCentersY[i-1];


        eyePatternFoundFlags[i]=eyePatternFoundFlags[i-1];
        scenePatternFoundFlags[i]=scenePatternFoundFlags[i-1];


    }

    //add new
    int timeElapsed = myTime.elapsed();


    timeStamps[0]= timeElapsed;
    eyePatternFoundFlags[0]=eyePatternFoundFlag;
    scenePatternFoundFlags[0]=scenePatternFoundFlag;

    eyePointsCentersX[0]=eyePointsCenterX;
    scenePointsCentersX[0]=scenePointsCenterX;

    eyePointsCentersY[0]=eyePointsCenterY;
    scenePointsCentersY[0]=scenePointsCenterY;



}









bool AutoGrabTwoCam:: pushAndCheck(
        bool eyePatternFoundFlag,
        bool scenePatternFoundFlag,
        double eyePointsCenterX,
        double scenePointsCenterX,
        double eyePointsCenterY,
        double scenePointsCenterY


        )
{

    shift2left(
                eyePatternFoundFlag,
                scenePatternFoundFlag,
                eyePointsCenterX,
                scenePointsCenterX,
                eyePointsCenterY,
                scenePointsCenterY
                );



    frameCount++;

    // step 1, check if the buffer contains enough frames aroud one second
    indexBound = 0; // default
    for (int i=1; i<N; i++){
        if (timeStamps[i]==0) return false;
        if (timeStamps[0]-timeStamps[i]>TIMESPAN){
            indexBound = i;
            break;
        }
    }
    if (indexBound<INDEX_BOUND_THRESHOLD) return false; // no enough frames collected
    frameRate = indexBound;


    // now, buffer[0, 1, ..., indexBuond] correspond to frames around 1 second
    // next, check if all contains patterns
    for (unsigned int i=0; i<=indexBound; i++){
        if(! eyePatternFoundFlags[i]) return false; // some frames contain no pattern
        if(! scenePatternFoundFlags[i]) return false; // some frames contain no pattern
    }

    // finally, check if frames move rapidly

    // compute mean
    eye_mx=0; scene_mx =0;
    eye_my=0; scene_my =0;

    for (unsigned int i=0; i<=indexBound; i++){
        eye_mx+=eyePointsCentersX[i];
        eye_my+=eyePointsCentersY[i];
        scene_mx+=scenePointsCentersX[i];
        scene_my+=scenePointsCentersY[i];
    }
    eye_mx/=(indexBound+1);
    eye_my/=(indexBound+1);
    scene_mx/=(indexBound+1);
    scene_my/=(indexBound+1);


    // compute sum of squares
    eye_ssx=0;
    eye_ssy=0;
    scene_ssx=0;
    scene_ssy=0;
    for (unsigned int i=0; i<=indexBound; i++){
        eye_ssx+=(eyePointsCentersX[i]-eye_mx)*(eyePointsCentersX[i]-eye_mx);
        eye_ssy+=(eyePointsCentersY[i]-eye_my)*(eyePointsCentersY[i]-eye_my);
        scene_ssx+=(scenePointsCentersX[i]-scene_mx)*(scenePointsCentersX[i]-scene_mx);
        scene_ssy+=(scenePointsCentersY[i]-scene_my)*(scenePointsCentersY[i]-scene_my);
    }


    // std deviation
    eye_stdx = sqrt(eye_ssx/(indexBound+1));
    eye_stdy = sqrt(eye_ssy/(indexBound+1));
    scene_stdx = sqrt(scene_ssx/(indexBound+1));
    scene_stdy = sqrt(scene_ssy/(indexBound+1));


    // square root and compare
    if (eye_stdx<THRESHOLD &&
            eye_stdy<THRESHOLD &&
            scene_stdx<THRESHOLD &&
            scene_stdy<THRESHOLD
            )     {
        clearTimeStamps(); // start over again
        goodImagePairCount++;
        return true;
    }else
        return false;
}












void AutoGrabTwoCam::writeInfoOnImage(cv::Mat &eyeImg, cv::Mat & sceneImg)
{
    QString qStr ;
    std::string stdStr;

    // show center of mass
    /*
    cv::circle(img,
               cv::Point(int(pointsCentersX[0]),int(pointsCentersY[1])),
               10,
               cv::Scalar(255,255,255),
               -1,
               8
               );

    */

    // show std deviation
    qStr = "std deviation: ";
    qStr += "(" + QString::number(eye_stdx,'f',5) + ", " + QString::number(eye_stdy,'f',5) + ")";
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,20),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );
    qStr = "std deviation: ";
    qStr += "(" + QString::number(scene_stdx,'f',5) + ", " + QString::number(scene_stdy,'f',5) + ")";
    stdStr = qStr.toStdString();
    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,20),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );

    // show flags

    qStr = "...";
    for (int i=15; i>=0; i--) qStr+= eyePatternFoundFlags[i]?'1':'0';
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,50),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );

    qStr = "...";
    for (int i=15; i>=0; i--) qStr+= scenePatternFoundFlags[i]?'1':'0';
    stdStr = qStr.toStdString();
    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,50),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );


    // show frameIndex
    qStr = "frame index = " + QString::number(frameCount);
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,80),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );
    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,80),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );


    // show elapsed time

    qStr = "time in seconds: = " + QString::number(timeStamps[0]/1000);
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,110),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );
    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,110),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );


    // show good image counts

    qStr = "# of auto shots: = " + QString::number(goodImagePairCount);
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,140),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );
    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,140),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );

    // show frame rate

    qStr = "ROUGH frame rate: = " + QString::number(frameRate);
    stdStr = qStr.toStdString();
    cv::putText(eyeImg,
                stdStr,
                cv::Point(20,170),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );

    cv::putText(sceneImg,
                stdStr,
                cv::Point(20,170),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );




}



























