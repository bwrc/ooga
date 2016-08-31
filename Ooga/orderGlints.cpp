//#include <opencv2/opencv.hpp>

#include "orderGlints.h"

std::vector<cv::Point2d> orderGlints(std::vector<cv::Point2d> glintPoints) {

  /* The glints are assumed to be shaped like this:
     
        1
    2       6
    3       5
        4

   */

	try{

		std::vector<cv::Point2d> glintPoints_ordered;
		int candidate_ind;
		int N_leds = glintPoints.size();
		if (N_leds != 6) {
			std::cout << "This will work only with 6 leds. Quitting." << std::endl;
			exit(-1);
		}
		//int ordered_inds[N_leds];
		int *ordered_inds = new int[N_leds]; //KL does this work?, note that this will be deleted in the end to avoid leaks
		//also, if only works with 6 leds (see check above), then why the dynamics?
		cv::Mat glints_x(N_leds, 1, CV_32F);
		cv::Mat glints_y(N_leds, 1, CV_32F);
		for (int i = 0; i < N_leds; i++) {
			glints_x.at<float>(i) = glintPoints[i].x;
			glints_y.at<float>(i) = glintPoints[i].y;
		}

		// First glint is the glint with lowest y value
		double min_val, max_val;
		cv::Point min_loc, max_loc;
		minMaxLoc(glints_y, &min_val, &max_val, &min_loc, &max_loc);
		ordered_inds[0] = min_loc.y;

		// Second glint is the highest of the glints that are left and below to the first glint
		candidate_ind = -1;
		for (int i = 0; i<N_leds; i++) {
			if ((glints_y.at<float>(i) > glints_y.at<float>(ordered_inds[0])) & \
				(glints_x.at<float>(i) < glints_x.at<float>(ordered_inds[0]))) {
				if (candidate_ind < 0) { candidate_ind = i; }
				if (glints_y.at<float>(i) < glints_y.at<float>(candidate_ind)) { candidate_ind = i; }
			}
		}
		ordered_inds[1] = candidate_ind;

		// Third glint is the leftmost of the glints that are below the second glint
		candidate_ind = -1;
		for (int i = 0; i<N_leds; i++) {
			if (glints_y.at<float>(i) > glints_y.at<float>(ordered_inds[1])) {
				if (candidate_ind < 0) { candidate_ind = i; }
				if (glints_x.at<float>(i) < glints_x.at<float>(candidate_ind)) { candidate_ind = i; }
			}
		}
		ordered_inds[2] = candidate_ind;

		// Fourth glint is the leftmost of the glints that are below the third glint
		candidate_ind = -1;
		for (int i = 0; i<N_leds; i++) {
			if (glints_y.at<float>(i) > glints_y.at<float>(ordered_inds[2])) {
				if (candidate_ind < 0) { candidate_ind = i; }
				if (glints_x.at<float>(i) < glints_x.at<float>(candidate_ind)) { candidate_ind = i; }
			}
		}
		ordered_inds[3] = candidate_ind;

		// Fifth glint is the lowest of the glints that are right the fourth glint
		candidate_ind = -1;
		for (int i = 0; i<N_leds; i++) {
			if (glints_x.at<float>(i) > glints_x.at<float>(ordered_inds[3])) {
				if (candidate_ind<0) { candidate_ind = i; }
				if (glints_y.at<float>(i) > glints_y.at<float>(candidate_ind)) { candidate_ind = i; }
			}
		}
		ordered_inds[4] = candidate_ind;

		// Infer the last glint
		for (int i = 0; i<N_leds; i++) {
			bool this_i_is_last_glint_ind = true;
			for (int j = 0; j<N_leds - 1; j++) {
				if (i == ordered_inds[j]) {  // This was one of the already ordered glints
					this_i_is_last_glint_ind = false;
					break;
				}
			}
			if (this_i_is_last_glint_ind) {  // This was NOT one of the already ordered glints --> it must be the last, un-ordered glint
				ordered_inds[5] = i;
			}
		}

		// Check that there are no negative indices left
		for (int i = 0; i < N_leds; i++) {
			if (ordered_inds[i] == -1) { ordered_inds[i] = i; }
		}

		// Form the ordered glint vector
		for (int i = 0; i < N_leds; i++) {
			glintPoints_ordered.push_back(glintPoints[ordered_inds[i]]);
		}

		delete[] ordered_inds; //KL del this if init with new

		return glintPoints_ordered;

	}
	catch (const std::exception& e){
		std::cout << "Glints could not be ordered " << e.what() << std::endl;
		return glintPoints;
		
	}

}
