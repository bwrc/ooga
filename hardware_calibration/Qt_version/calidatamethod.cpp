#include "calidatamethod.h"
#include "opencv2/opencv.hpp"
#include <QtGlobal>


void CaliDataMethod::setNumOfImagePairs(unsigned int x){
   // Q_ASSERT(x<50);

    noip = x;
}

CaliDataMethod::CaliDataMethod()
{

    cvObjectPointsSmallBoard.clear();
    cvObjectPointsBigBoard.clear();

    // make points for one view, and a vector of multiple views
    for (int i=0; i<11; i++) // outer iterate, row by row
        for (int j=0; j<4; j++) // inner iteration, colum by colum
        {
            cvObjectPointsBigBoard.push_back(
                        cv:: Point3f( // floating point, check opencv code (x ja y oli toisin päin t:Miika)
                                      (float(10) - float(i*1)),  // tämä oli float(i*1),
                                      float((2*j + i % 2)*1),
                                      0
                                      )
                        );


        }
}


void CaliDataMethod:: setEyeCameraMatrix(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(3,3));
    Q_ASSERT(x.type()==CV_64FC1);
    Q_ASSERT(!x.empty());

    x.copyTo(eyeCameraMatrix);

}

void CaliDataMethod:: setEyeCameradistVector(const cv::Mat & x){

    Q_ASSERT(x.size()==cv::Size(1,5)); // .size() returns col x row
    Q_ASSERT(x.type()==CV_64FC1);
    Q_ASSERT(!x.empty());

    x.copyTo(eyeCameraDistVector);
}

void CaliDataMethod:: setSceneCameraMatrix(const cv::Mat & x){
    Q_ASSERT(x.size()==cv::Size(3,3));
    Q_ASSERT(x.type()==CV_64FC1);
    Q_ASSERT(!x.empty());

    x.copyTo(sceneCameraMatrix);

}

void CaliDataMethod:: setSceneCameraDistVector(const cv::Mat & x){

    Q_ASSERT(x.size()==cv::Size(1,5)); // .size() returns col x row
    Q_ASSERT(x.type()==CV_64FC1);
    Q_ASSERT(!x.empty());


    x.copyTo(sceneCameraDistVector);
}

void CaliDataMethod:: setRigMatrix(const cv::Mat & x){

    Q_ASSERT(x.size()==cv::Size(4,4));
    Q_ASSERT(x.type()==CV_64FC1);
    Q_ASSERT(!x.empty());

    x.copyTo(rigMatrix);

}

void CaliDataMethod:: setSf(double x){

    Q_ASSERT(x>0);
    Q_ASSERT(x<1);

    //x = 1; // special: only for computing the left-to-right eye transformation (comment otherwise!!)
    sf = x;

    for (int i=0; i<11; i++) // outer iterate, row by row
        for (int j=0; j<4; j++) // inner iteration, colum by colum
            cvObjectPointsSmallBoard.push_back(cv:: Point3f( // floating point required, check opencv source code (x ja y oli toisin päin t: Miika)
                                                             (float(10)-float(i*1)) * float(x),  // tämä oli "float(i*1) * float(x),"
                                                             float((2*j + i % 2)*1) * float(x),
                                                             0
                                                             )
                                               );




}

void CaliDataMethod:: setCvEyeCcsVector(const std::vector<std::vector<cv::Point2f> > & x){

    Q_ASSERT(x.size() == noip);

    cvEyeCcsVector = x;

}

void CaliDataMethod:: setCvSceneCcsVector(const std::vector<std::vector<cv::Point2f> > & x){
    Q_ASSERT(x.size() == noip);
    cvSceneCcsVector=x;

}


void CaliDataMethod::calibrate(){

    rmats.clear();
    tvecs.clear();

    for (unsigned int pairIndex=0; pairIndex<noip; pairIndex++){

      // std::cout << "cvObjectPointsSmallBoard:  " << std::endl << cvObjectPointsSmallBoard << std::endl;
      // std::cout << "cvEyeCcsVector[pairIndex]:  " << std::endl << cvEyeCcsVector[pairIndex] << std::endl;

        // from virtual eye cam to small board
        try {
            bool success = cv::solvePnP(cvObjectPointsSmallBoard,
                                        cvEyeCcsVector[pairIndex],
                                        eyeCameraMatrix,
                                        eyeCameraDistVector,
                                        rvecVirtualEyeCam2SmallBoard,
                                        tvecVirtualEyeCam2SmallBoard
                                        );

	    //std::cout << std::endl << "   tvec sb to ec:  " << std::endl << tvecVirtualEyeCam2SmallBoard << std::endl;
            if (!success) {
                std::cout << std::endl;
                std::cout << " -------------- solvePnP returned false, image pair number: " << pairIndex << std::endl;
                noip--;
                continue;
            }
        } catch (std::exception e) {
            std::cout << std::endl;
            std::cout << " -------------- solvePnP failed, image pair number: " << pairIndex << std::endl;
            noip--;
            continue;
        }

        Q_ASSERT(rvecVirtualEyeCam2SmallBoard.size()==cv::Size(1,3));
        // .size() returns cols x rows
        Q_ASSERT(tvecVirtualEyeCam2SmallBoard.size()==cv::Size(1,3));


        Q_ASSERT(rvecVirtualEyeCam2SmallBoard.type()==CV_64FC1);
        Q_ASSERT(tvecVirtualEyeCam2SmallBoard.type()==CV_64FC1);


        cv::Rodrigues(rvecVirtualEyeCam2SmallBoard,tmpM33);

        Q_ASSERT(tmpM33.size()==cv::Size(3,3));
        Q_ASSERT(tmpM33.type()==CV_64FC1);

        cvM44VirtualEyeCam2SmallBoard = cv::Mat:: eye(4,4, CV_64FC1);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                cvM44VirtualEyeCam2SmallBoard.at<double>(i,j)=tmpM33.at<double>(i,j);
        for (int i=0; i<3; i++)
            cvM44VirtualEyeCam2SmallBoard.at<double>(i,3)=tvecVirtualEyeCam2SmallBoard.at<double>(i,0);


        // from scene cam to big board
        try {
            bool success = cv::solvePnP(cvObjectPointsBigBoard,
                                        cvSceneCcsVector[pairIndex],
                                        sceneCameraMatrix,
                                        sceneCameraDistVector,
                                        rvecSceneCam2BigBoard,
                                        tvecSceneCam2BigBoard
                                        );

	    //std::cout << std::endl << "   tvec bb to sc:  " << std::endl << tvecSceneCam2BigBoard << std::endl;

            if (!success) {
                std::cout << std::endl;
                std::cout << " -------------- solvePnP returned false, image pair number: " << pairIndex << std::endl;
                noip--;
                continue;
            }
        } catch (std::exception e) {
            std::cout << std::endl;
            std::cout << " -------------- solvePnP failed, image pair number: " << pairIndex << std::endl;
            noip--;
            continue;
        }


        Q_ASSERT(rvecSceneCam2BigBoard.size()==cv::Size(1,3));
        // .size() returns cols x rows
        Q_ASSERT(tvecSceneCam2BigBoard.size()==cv::Size(1,3));


        Q_ASSERT(rvecSceneCam2BigBoard.type()==CV_64FC1);
        Q_ASSERT(tvecSceneCam2BigBoard.type()==CV_64FC1);

        // convert
        cv::Rodrigues(rvecSceneCam2BigBoard,tmpM33);

        Q_ASSERT(tmpM33.size()==cv::Size(3,3));
        Q_ASSERT(tmpM33.type()==CV_64FC1);

        cvM44SceneCam2BigBoard = cv::Mat:: eye(4,4, CV_64FC1);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                cvM44SceneCam2BigBoard.at<double>(i,j)=tmpM33.at<double>(i,j);
        for (int i=0; i<3; i++)
            cvM44SceneCam2BigBoard.at<double>(i,3)=tvecSceneCam2BigBoard.at<double>(i,0);


        // chnage direction, from big board to scene cam
        cv::invert(cvM44SceneCam2BigBoard,cvM44BigBoard2SceneCam)        ;
        Q_ASSERT(cvM44BigBoard2SceneCam.type()==CV_64FC1);




	// make products (BUT NOT LIKE THIS,  t: Miika)
	// cvM44VirtualEyeCam2BigBoard = cv::Mat::eye(4,4,CV_64FC1);
	// matrixProduct(cvM44VirtualEyeCam2SmallBoard,rigMatrix,cvM44VirtualEyeCam2BigBoard );

	// cvM44VirtualEyeCam2SceneCam = cv::Mat::eye(4,4,CV_64FC1);
	// matrixProduct(cvM44VirtualEyeCam2BigBoard,cvM44BigBoard2SceneCam,cvM44VirtualEyeCam2SceneCam);


	// -------------------------
	// Matrix product was wrong way?

	//std::cout << std::endl << std::endl << "Small board to Eye camera: " << std::endl << cvM44VirtualEyeCam2SmallBoard << std::endl << std::endl;

	cv::Mat cvM44EyeCam2BigBoard = cv::Mat::eye(4,4,CV_64FC1);
	cv::Mat rigMatrix_inv;
	cv::invert(cvM44VirtualEyeCam2SmallBoard, cvM44VirtualEyeCam2SmallBoard);  // this was wrong way (board to cam)
	matrixProduct(rigMatrix, cvM44VirtualEyeCam2SmallBoard, cvM44EyeCam2BigBoard );

	cvM44VirtualEyeCam2SceneCam = cv::Mat::eye(4,4,CV_64FC1);
	cv::invert(cvM44BigBoard2SceneCam, cvM44BigBoard2SceneCam);  // this was wrong way (cam to board)
	matrixProduct(cvM44BigBoard2SceneCam, cvM44EyeCam2BigBoard, cvM44VirtualEyeCam2SceneCam);

	// std::cout << std::endl << std::endl << "Eye camera to Small board: " << std::endl << cvM44VirtualEyeCam2SmallBoard << std::endl << std::endl;
	// std::cout << std::endl << std::endl << "Small board to Big board: " << std::endl << rigMatrix << std::endl << std::endl;
	// std::cout << std::endl << std::endl << "Big board to Scene camera: " << std::endl << cvM44BigBoard2SceneCam << std::endl << std::endl;
	// std::cout << std::endl << std::endl << "---> Eye camera to Scene camera: " << std::endl << cvM44VirtualEyeCam2SceneCam << std::endl << std::endl;
	// std::cout << std::endl << std::endl << " __________________________  " << std::endl << std::endl << std::endl;

	// -------------------------


        if (1) {
            std::cout << std::endl << "image pair " << pairIndex << ": m44 as product: "
                     << cvM44VirtualEyeCam2SceneCam;
        }


        // tmp varialbes
        for(int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                tmpM33.at<double>(i,j)=cvM44VirtualEyeCam2SceneCam.at<double>(i,j);
        tvecVirtualEyeCam2SceneCam = cv::Mat::zeros(3,1,CV_64FC1);
        for(int i=0; i<3; i++)
            tvecVirtualEyeCam2SceneCam.at<double>(i,0) = cvM44VirtualEyeCam2SceneCam.at<double>(i,3);

        // push to vector
        rmats.push_back(tmpM33.clone()); // comment 119: a bug if no use of clone, check comment 120
        tvecs.push_back(tvecVirtualEyeCam2SceneCam.clone());
    }


    //take averge over, tvecs, attention: same procedure not applicable to rvec
    cv::Mat tvecSum = cv::Mat::zeros(3,1,CV_64FC1);
    for (unsigned int i=0; i<noip; i++)       tvecSum += tvecs[i];
    tvecVirtualEyeCam2SceneCam = tvecSum/noip;

    // --- (I wonder it the following is correct? Perhaps average in the quaternion space instead? t: Miika)
    //take averge over rvecs, attention, another method needed, average over 3x3 matrix, and not
    // using z-axis, using x and y axies only
    cv::Mat rmatSum = cv::Mat::zeros(3,3,CV_64FC1)    ;
    for (unsigned int i=0; i<noip; i++)     rmatSum += rmats[i];  // this looks bad, averaging the rotation matrices... (t: Miika)
    cv::Mat rmatSumAverage = rmatSum / noip;
    // realtion of variables in the following:
    /*
    */
    // now, first column is x asix, and 2nd is y
    cv::Mat x_old  = rmatSumAverage.col(0);
    cv::Mat y_old  = rmatSumAverage.col(1);
    cv::Mat z_new = x_old.cross(y_old);
    cv::Mat mean_x_y_old = (x_old + y_old) ; // this part needs doc
    mean_x_y_old = mean_x_y_old / cv::norm(mean_x_y_old);
    cv::Mat xx = mean_x_y_old.cross(z_new);
    xx = xx / cv::norm(xx);
    cv::Mat yy = -xx;
    cv::Mat x_new = ( mean_x_y_old + xx ) ;
    x_new = x_new / cv::norm(x_new);
    cv::Mat y_new = ( mean_x_y_old + yy ) ;
    y_new = y_new / cv::norm(y_new);
    z_new = x_new.cross(y_new);
    z_new = z_new / cv::norm(z_new);





    // final update
    cvM44VirtualEyeCam2SceneCam = cv::Mat::eye(4,4,CV_64FC1);
    //
    cvM44VirtualEyeCam2SceneCam.at<double>(0,0) = x_new.at<double>(0);
    cvM44VirtualEyeCam2SceneCam.at<double>(1,0) = x_new.at<double>(1);
    cvM44VirtualEyeCam2SceneCam.at<double>(2,0) = x_new.at<double>(2);
    //
    cvM44VirtualEyeCam2SceneCam.at<double>(0,1) = y_new.at<double>(0);
    cvM44VirtualEyeCam2SceneCam.at<double>(1,1) = y_new.at<double>(1);
    cvM44VirtualEyeCam2SceneCam.at<double>(2,1) = y_new.at<double>(2);
    //
    cvM44VirtualEyeCam2SceneCam.at<double>(0,2) = z_new.at<double>(0);
    cvM44VirtualEyeCam2SceneCam.at<double>(1,2) = z_new.at<double>(1);
    cvM44VirtualEyeCam2SceneCam.at<double>(2,2) = z_new.at<double>(2);

    for (int i=0; i<3; i++)
        cvM44VirtualEyeCam2SceneCam.at<double>(i,3)=tvecVirtualEyeCam2SceneCam.at<double>(i,0);



    if (1) {
      std::cout << std::cout << "Average over " << noip << " elements. " << std::endl;
      std::cout << std::endl << std::endl << "m44 after average: " << cvM44VirtualEyeCam2SceneCam;
    }
}

void CaliDataMethod:: matrixProduct(cv::Mat & a, cv::Mat & b, cv::Mat & c){
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            c.at<double>(i,j)=
                    a.at<double>(i,0) * b.at<double>(0,j) +
                    a.at<double>(i,1) * b.at<double>(1,j) +
                    a.at<double>(i,2) * b.at<double>(2,j) +
                    a.at<double>(i,3) * b.at<double>(3,j) ;
}
