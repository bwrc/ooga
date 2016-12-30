#include "autograb.h"



AutoGrab::AutoGrab()
{

    ini();
}

void AutoGrab::clearTimeStamps(){
    for (int i=0; i<N; i++) timeStamps[i]=0;
}

void AutoGrab::ini(){

    clearTimeStamps();

    frameCount = 0;
    goodImageCount = 0;

    myTime.start();

}

void AutoGrab::shift2left(
        //int timeElapsed,
        bool patternFoundFlag,
        double pointsCenterX,
        double pointsCenterY)
{
    for (int i=N-1; i>0; i--){
        timeStamps[i]=timeStamps[i-1];
        pointsCentersX[i]=pointsCentersX[i-1];
        pointsCentersY[i]=pointsCentersY[i-1];
        patternFoundFlags[i]=patternFoundFlags[i-1];
    }

    //add new
    int timeElapsed = myTime.elapsed();
    //qDebug()<<"\nTime Elapsed: "<<timeElapsed;

    timeStamps[0]= timeElapsed;
    patternFoundFlags[0]=patternFoundFlag;
    pointsCentersX[0]=pointsCenterX;
    pointsCentersY[0]=pointsCenterY;
}



bool AutoGrab:: pushAndCheck(
        //int timeElapsed,
        bool patternFoundFlag,
        double pointsCenterX, double pointsCenterY)
{

    shift2left(
                //timeElapsed,
                patternFoundFlag, pointsCenterX, pointsCenterY);
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
    //qDebug()<<"\n::pushAndCheck::indexBound: "<<indexBound;
    if (indexBound<INDEX_BOUND_THRESHOLD) return false; // no enough frames collected
    frameRate = indexBound;


    // now, buffer[0, 1, ..., indexBuond] correspond to frames around 1 second
    // next, check if all contains patterns
    for (unsigned int i=0; i<=indexBound; i++){
        if(! patternFoundFlags[i]) return false; // some frames contain no pattern
    }

    // finally, check if frames move rapidly

    // compute mean
    mx=0;
    my=0;
    //qDebug()<<"\n";
    for (unsigned int i=0; i<=indexBound; i++){
        mx+=pointsCentersX[i];
        my+=pointsCentersY[i];


        //qDebug()<<i<<": ("<< pointsCentersX[i]<<","<<pointsCentersY[i]<<"), ";
    }
    mx/=(indexBound+1);
    my/=(indexBound+1);
    //qDebug()<<"\nmx: "<<mx<<"\nmy: "<<my;

    // compute sum of squares
    ssx=0;
    ssy=0;
    for (unsigned int i=0; i<=indexBound; i++){
        ssx+=(pointsCentersX[i]-mx)*(pointsCentersX[i]-mx);
        ssy+=(pointsCentersY[i]-my)*(pointsCentersY[i]-my);
    }
    //qDebug()<<"\nssx: "<<ssx<<"\nssy: "<<ssy;

    // std deviation
    stdx = sqrt(ssx/(indexBound+1));
    stdy = sqrt(ssy/(indexBound+1));
    //qDebug()<<"\nstdx: "<<stdx<<"\nstdy: "<<stdy;



    // square root and compare
    if (stdx<THRESHOLD && stdy<THRESHOLD)     {
        clearTimeStamps(); // start over again
        goodImageCount++;
        return true;
    }else
        return false;
}






void AutoGrab::writeInfoOnImage(cv::Mat &img){
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
    qStr += "(" + QString::number(stdx,'f',5) + ", " + QString::number(stdy,'f',5) + ")";
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,20),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );


    // show flags
    qStr.clear();
    qStr = "...";
    for (int i=15; i>=0; i--) qStr+= patternFoundFlags[i]?'1':'0';
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,50),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );



    // show frameIndex
    qStr = "frame index = " + QString::number(frameCount);
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,80),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );



    // show elapsed time

    qStr = "time in seconds: = " + QString::number(timeStamps[0]/1000);
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,110),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );



    // show good image counts

    qStr = "# of auto shots: = " + QString::number(goodImageCount);
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,140),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );


    // show frame rate

    qStr = "ROUGH frame rate: = " + QString::number(frameRate);
    stdStr = qStr.toStdString();
    cv::putText(img,
                stdStr,
                cv::Point(20,170),
                cv::FONT_HERSHEY_DUPLEX,
                0.9,
                cv::Scalar(255,255,255)
                );





}
