#include "PupilEstimator.h"

#include <stdlib.h> //min, max

PupilEstimator::PupilEstimator()
{
}


PupilEstimator::~PupilEstimator()
{
}



void PupilEstimator::Initialize()
{

}

//cv::RotatedRect PupilEstimator::getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element)
cv::RotatedRect PupilEstimator::getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta)
{

	ITERATE = pupil_iterate;

	float Svert = eyeImage_opened.rows;
	float Shori = eyeImage_opened.cols;
	cv::Point2d pupil_center_in = pupil_center;

	// Crop the image around the pupil center
	x_min = std::max(float(0), float(pupil_center.x) - delta_x);
	x_max = std::min(Shori, float(pupil_center.x) + delta_x);
	y_min = std::max(float(0), float(pupil_center.y) - delta_y);
	y_max = std::min(Svert, float(pupil_center.y) + delta_y);
	// Miksi ei kropata t‰ss‰?! t: Miika
	//eyeImage_cropped = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)); //t‰‰ on laillisuuden rajamailla, funktiokutsun mukaan ton ei pit‰is muokkaantua, 
	//mutta nyt croppedin modaaminen muokkaisi sit‰. T‰‰ tosin vaan filtterˆid‰‰n, joten koko cropped on turha?
	pupil_center.x = pupil_center.x - x_min;   // Pupil center in the cropped image
	pupil_center.y = pupil_center.y - y_min;   // Pupil center in the cropped image

	// Filter the image (and crop away edges)
	cv::filter2D(eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)), // <- was eyeImage_cropped
		     eyeImage_filtered, -1, pupil_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

	int delta_edge = 10;  // Crop the edges away
	if (delta_edge > delta_x || delta_edge > delta_y) { std::cout << "delta_edge is larger than the image half " << std::endl; exit(-1); }
	
	//cv::Mat eyeImage_filtered_cropped = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge));
	morphologyEx(eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)),
		     eyeImage_closed, cv::MORPH_CLOSE, pupil_element);
	eyeImage_sum = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)) + eyeImage_closed; //cv::add might use optimized code?
	pupil_center.x = pupil_center.x - delta_edge;   // Pupil center (x) in the cropped image
	pupil_center.y = pupil_center.y - delta_edge;   // Pupil center (y) in the cropped image

	// Scale the image
	//avoid conversion operations by doing it in 8-bit space?
	/*
		eyeImage_sum.convertTo(eyeImage_sum_float, CV_32F);
		cv::normalize(eyeImage_sum_float, eyeImage_sum_scaled, 0, 1, cv::NORM_MINMAX, -1);

		// Binarize the image
		float thold = 0.15;  // was 0.15
		threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 1, cv::THRESH_BINARY_INV);

		// Get the connected components of the binary image
		eyeImage_binary.convertTo(eyeImage_binary_8bit, CV_8UC1);
	*/
	int thold = floor(0.15 * 255);  // TODO: as input parameter (read it from file)
	cv::normalize(eyeImage_sum, eyeImage_sum_scaled, 0, 255, cv::NORM_MINMAX, -1);
	cv::threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 255, cv::THRESH_BINARY_INV);
	//the replacement for the commented bit above ends here
	//cv::connectedComponents(eyeImage_binary_8bit, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)
	cv::connectedComponents(eyeImage_binary, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)

	// Get the pupil blob, being the area which encloses the pupil center
	int pupilLabel = connComps.at<int>(pupil_center.y, pupil_center.x);  // index of the pupil area
	connComps.convertTo(connComps_32F, CV_32F);
	//KL <- muutettu pysyv‰ksi cv::Mat connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	threshold(connComps_aux, pupil_blob, 0.5, 1, cv::THRESH_BINARY_INV);  // get the pupil area

	// Find the contours of the pupil blob
	pupil_blob.convertTo(pupil_blob_8bit, CV_8U);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(pupil_blob_8bit, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	//select the longest contour
	pupil_contour = contours[0];
	for (int i = 1; i<contours.size(); i++) {
		if (contours[i].size() > pupil_contour.size()) {
			pupil_contour = contours[i];
		}
	}

	//not a very neat solution?
	while (pupil_contour.size() < 6) {  // If pupil_contour happens to contain less than 5 points (this shouldn't normally happen), add the first point until N>5
		pupil_contour.push_back(pupil_contour[0]);
	}

	if (!ITERATE) {
		// Compute the pupil_contour in the original (incoming) image coordinates
		for (int i = 0; i<pupil_contour.size(); i++) {
			pupil_contour[i].x = pupil_contour[i].x + x_min + delta_edge;
			pupil_contour[i].y = pupil_contour[i].y + y_min + delta_edge;
		}
		cv::RotatedRect pupilEllipse = cv::fitEllipse(pupil_contour);
		//return pupilEllipse;
		return cv::RotatedRect(cv::fitEllipse(pupil_contour));
		//TODO: this results in an assertion failure _block_type_is_valid(pHead->nBlockUse)
		//THIS IS PROBABLY DUE TO RETURNING AN OBJECT FROM AN OBJECT, WHICH IS ALLOCATED ON THE STACK AND GOES OUT OF SCOPE WHEN THE FUNCTION ENDS
	} 
	
	else {

		//  %%%%%%%%%%%%%%%%%%%%%  FINAL ITERATION  %%%%%%%%%%%%%%%%%%%%%%%%

		// 	eyeImage_clone = eyeImage_cropped.clone();
		eyeImage_clone = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone();

		//removed the need for cropped -> use the clone
		//eyeImage_cropped.convertTo(eyeImage_cropped, CV_32F);
		//cv::normalize(eyeImage_cropped, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);
		eyeImage_clone.convertTo(eyeImage_clone, CV_32F);
		cv::normalize(eyeImage_clone, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);

		// Compute the pupil_contour in the cropped image coordinates
		for (int i = 0; i < pupil_contour.size(); i++) {
			pupil_edge.push_back(cv::Point2d(pupil_contour[i].x + delta_edge, pupil_contour[i].y + delta_edge));
			// circle(eyeImage_clone, cv::Point2d(pupil_contour[i].x + delta_edge , pupil_contour[i].y + delta_edge) , 2, 250, -1, 8);
		}

		cv::Mat edge_filt = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, 1, -1);

		cv::Sobel(1 - eyeImage_norm, imFilt_up, -1, 0, 1);
		cv::Sobel(eyeImage_norm, imFilt_down, -1, 0, 1);
		cv::Sobel(1 - eyeImage_norm, imFilt_left, -1, 1, 0);
		cv::Sobel(eyeImage_norm, imFilt_right, -1, 1, 0);

		cv::normalize(imFilt_up, imFilt_up, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_down, imFilt_down, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_left, imFilt_left, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_right, imFilt_right, 0, 1, cv::NORM_MINMAX, -1);

		float beta = pupil_beta;// 20;
		int n = 1;
		int post_val_prev = 0;
		double postDiff_thold = 0;
		double postDiff_thold_weight = 0.1;
		int maxN = 10;
		int x1, x2, y1, y2;
		float delta = 15;

		cv::Mat log_likelihood;  //KL are these big?
		double mean_log_MAP_prev = 0;
		cv::RotatedRect ellipse;
		cv::Mat imFilt_crop;

		while (1) {
			//cout << n << endl;
			cv::Mat eyeImage_clone2 = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone(); //eyeImage_clone.clone();

			if (0) { // PLOT
				for (int i = 0; i < pupil_edge.size(); i++) {
					circle(eyeImage_clone2, pupil_edge[i], 2, 150, -1, 8);
					imshow("kuva", eyeImage_clone2);
				}
				cv::waitKey(0);
			}

			ellipse = cv::fitEllipse(pupil_edge);
			pupil_edge.clear();

			float a = ellipse.size.height / 2;
			float b = ellipse.size.width / 2;
			float phi = (270 - ellipse.angle) *CV_PI / 180;
			float Xc = ellipse.center.x;
			float Yc = ellipse.center.y;
			std::vector<cv::Point2f> ellipse_points;
			double angle_step = 0.3; // 2*CV_PI / angle_step > 5 must hold !
			for (float t = 0; t < 2 * CV_PI; t = t + angle_step) {
			  float ellipse_point_X = std::min(std::max(float(0), Xc + a*std::cos(t)*std::cos(phi) - b*std::sin(t)*std::sin(phi)), float(x_max - x_min)); // float(eyeImage_cropped.cols)); KL - this should be the same number, eh?
			  float ellipse_point_Y = std::min(std::max(float(0), Yc + a*std::cos(t)*std::sin(phi) + b*std::sin(t)*std::cos(phi)), float(y_max - y_min)); // float(eyeImage_cropped.rows));
				ellipse_points.push_back(cv::Point2f(ellipse_point_X, ellipse_point_Y));
				//ellipse_points.push_back(cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)));
				// circle(eyeImage_clone2,  cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)), 2, 150, -1, 8);
			}
			// imshow("kuva", eyeImage_clone2);
			// cv::waitKey(1);

			std::vector<double> log_MAP;
			for (int i = 0; i < ellipse_points.size(); i++) {

				x1 = std::max(float(0), ellipse_points[i].x - delta);
				x2 = std::min(float(x_max-x_min/*eyeImage_cropped.cols*/), ellipse_points[i].x + delta);
				y1 = std::max(float(0), ellipse_points[i].y - delta);
				y2 = std::min(float(y_max-y_min/*eyeImage_cropped.rows*/), ellipse_points[i].y + delta);

				if (Yc - ellipse_points[i].y >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the upper sector of the pupil
					imFilt_crop = imFilt_up(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the eye lid often produces a disturbing edge --> divide beta by half for the upper sector of the pupil
				}
				if (ellipse_points[i].y - Yc >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the bottom sector of the pupil
					imFilt_crop = imFilt_down(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the lower eye lid often produces a disturbing edge --> scale beta by 0.75 for the lower sector of the pupil
				}
				if (Xc - ellipse_points[i].x >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the left sector of the pupil
					imFilt_crop = imFilt_left(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}
				if (ellipse_points[i].x - Xc >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the right sector of the pupil
					imFilt_crop = imFilt_right(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}

				log_likelihood = beta * imFilt_crop;

				// Form the coordinates for which to evaluate the prior
				// perhaps Eigen would be quicker here?
				cv::Mat xv(1, x2 - x1, CV_32F);
				cv::Mat yv(y2 - y1, 1, CV_32F);
				int ii = 0; for (int x = x1; x < x2; x++) { xv.at<float>(0, ii++) = x; }
				ii = 0;     for (int y = y1; y < y2; y++) { yv.at<float>(ii++, 0) = y; }
				cv::Mat xx = repeat(xv, yv.total(), 1);
				cv::Mat yy = repeat(yv, 1, xv.total());
				cv::Mat xcoords = xx.reshape(0, 1);
				cv::Mat ycoords = yy.reshape(0, 1);

				cv::Mat dist_x2, dist_y2;
				cv::pow(ellipse_points[i].x - xcoords, 2, dist_x2);
				cv::pow(ellipse_points[i].y - ycoords, 2, dist_y2);
				cv::Mat dist_vec;
				cv::sqrt(dist_x2 + dist_y2, dist_vec);
				cv::Mat dist = dist_vec.reshape(0, yv.rows);
				cv::Mat log_prior = -dist;
				cv::Mat log_post = log_likelihood + log_prior;

				minMaxLoc(log_post, &min_val, &max_val, &min_loc, &max_loc);

				max_loc.x = max_loc.x + x1;
				max_loc.y = max_loc.y + y1;

				pupil_edge.push_back(max_loc);
				log_MAP.push_back(max_val);
			}

			if (n == 2) {
				postDiff_thold = std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) * postDiff_thold_weight;
			}  // get the threshold automatically

			//while gets repeated until 
			if (n++ > maxN || (std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) < postDiff_thold)) {
				break;
			}

			mean_log_MAP_prev = cv::mean(log_MAP)[0];
			log_MAP.clear();

		}  // end of while(1)


		if (0) {  // PLOT
			for (int i = 0; i < pupil_edge.size(); i++) {
				circle(eyeImage_clone, pupil_edge[i], 2, 150, -1, 8);
				imshow("Lopullinen kuva", eyeImage_clone);
			}
			cv::waitKey(1);
		}

		ellipse = cv::fitEllipse(pupil_edge);
		ellipse.center.x = ellipse.center.x + x_min;
		ellipse.center.y = ellipse.center.y + y_min;
		ellipse.angle = (270 - ellipse.angle) *CV_PI / 180;

		return ellipse;
	}
}

void PupilEstimator::getPupilEllipse(cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element, bool pupil_iterate, float pupil_beta, cv::RotatedRect &pupilEllipse)
{

  // Onko t‰m‰ funktio turha? t: Miika

	ITERATE = pupil_iterate;

	float Svert = eyeImage_opened.rows;
	float Shori = eyeImage_opened.cols;
	cv::Point2d pupil_center_in = pupil_center;

	// Crop the image around the pupil center
	x_min = std::max(float(0), float(pupil_center.x) - delta_x);
	x_max = std::min(Shori, float(pupil_center.x) + delta_x);
	y_min = std::max(float(0), float(pupil_center.y) - delta_y);
	y_max = std::min(Svert, float(pupil_center.y) + delta_y);
	//eyeImage_cropped = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)); //t‰‰ on laillisuuden rajamailla, funktiokutsun mukaan ton ei pit‰is muokkaantua, 
	//mutta nyt croppedin modaaminen muokkaisi sit‰. T‰‰ tosin vaan filtterˆid‰‰n, joten koko cropped on turha?
	pupil_center.x = pupil_center.x - x_min;   // Pupil center in the cropped image
	pupil_center.y = pupil_center.y - y_min;   // Pupil center in the cropped image

	// Filter the image (and crop away edges)
	cv::filter2D(eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)), // <- was eyeImage_cropped
		eyeImage_filtered, -1, pupil_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

	int delta_edge = 10;  // Crop the edges away
	if (delta_edge > delta_x || delta_edge > delta_y) { std::cout << "delta_edge is larger than the image half " << std::endl; exit(-1); }

	//cv::Mat eyeImage_filtered_cropped = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge));
	morphologyEx(eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)),
		     eyeImage_closed, cv::MORPH_CLOSE, pupil_element);
	eyeImage_sum = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)) + eyeImage_closed; //cv::add might use optimized code?
	pupil_center.x = pupil_center.x - delta_edge;   // Pupil center (x) in the cropped image
	pupil_center.y = pupil_center.y - delta_edge;   // Pupil center (y) in the cropped image

	// Scale the image
	//avoid conversion operations by doing it in 8-bit space?
	/*
	eyeImage_sum.convertTo(eyeImage_sum_float, CV_32F);
	cv::normalize(eyeImage_sum_float, eyeImage_sum_scaled, 0, 1, cv::NORM_MINMAX, -1);

	// Binarize the image
	float thold = 0.15;  // was 0.15
	threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 1, cv::THRESH_BINARY_INV);

	// Get the connected components of the binary image
	eyeImage_binary.convertTo(eyeImage_binary_8bit, CV_8UC1);
	*/
	int thold = floor(0.15 * 255);
	cv::normalize(eyeImage_sum, eyeImage_sum_scaled, 0, 255, cv::NORM_MINMAX, -1);
	cv::threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 255, cv::THRESH_BINARY_INV);
	//the replacement for the commented bit above ends here
	//cv::connectedComponents(eyeImage_binary_8bit, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)
	cv::connectedComponents(eyeImage_binary, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)

	// Get the pupil blob, being the area which encloses the pupil center
	int pupilLabel = connComps.at<int>(pupil_center.y, pupil_center.x);  // index of the pupil area
	connComps.convertTo(connComps_32F, CV_32F);
	//KL <- muutettu pysyv‰ksi cv::Mat connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	threshold(connComps_aux, pupil_blob, 0.5, 1, cv::THRESH_BINARY_INV);  // get the pupil area

	// Find the contours of the pupil blob
	pupil_blob.convertTo(pupil_blob_8bit, CV_8U);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(pupil_blob_8bit, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	//select the longest contour
	pupil_contour = contours[0];
	for (int i = 1; i<contours.size(); i++) {
		if (contours[i].size() > pupil_contour.size()) {
			pupil_contour = contours[i];
		}
	}

	//not a very neat solution?
	while (pupil_contour.size() < 6) {  // If pupil_contour happens to contain less than 5 points (this shouldn't normally happen), add the first point until N>5
		pupil_contour.push_back(pupil_contour[0]);
	}

	if (!ITERATE) {
		// Compute the pupil_contour in the original (incoming) image coordinates
		for (int i = 0; i<pupil_contour.size(); i++) {
			pupil_contour[i].x = pupil_contour[i].x + x_min + delta_edge;
			pupil_contour[i].y = pupil_contour[i].y + y_min + delta_edge;
		}
		//cv::RotatedRect pupilEllipse = cv::fitEllipse(pupil_contour);
		//return pupilEllipse;
		//return cv::RotatedRect(cv::fitEllipse(pupil_contour));
		//TODO: this results in an assertion failure _block_type_is_valid(pHead->nBlockUse)
		//OOF NEXT IDEA: THIS IS BECAUSE cv::Mat's or other things are being reused and pics of different sizes/content get in there
		pupilEllipse = cv::fitEllipse(pupil_contour); //should copy this to the ref'd param
	}

	else {

		//  %%%%%%%%%%%%%%%%%%%%%  FINAL ITERATION  %%%%%%%%%%%%%%%%%%%%%%%%

		// 	eyeImage_clone = eyeImage_cropped.clone();
		eyeImage_clone = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone();

		//removed the need for cropped -> use the clone
		//eyeImage_cropped.convertTo(eyeImage_cropped, CV_32F);
		//cv::normalize(eyeImage_cropped, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);
		eyeImage_clone.convertTo(eyeImage_clone, CV_32F);
		cv::normalize(eyeImage_clone, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);

		// Compute the pupil_contour in the cropped image coordinates
		for (int i = 0; i < pupil_contour.size(); i++) {
			pupil_edge.push_back(cv::Point2d(pupil_contour[i].x + delta_edge, pupil_contour[i].y + delta_edge));
			// circle(eyeImage_clone, cv::Point2d(pupil_contour[i].x + delta_edge , pupil_contour[i].y + delta_edge) , 2, 250, -1, 8);
		}

		cv::Mat edge_filt = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, 1, -1);

		cv::Sobel(1 - eyeImage_norm, imFilt_up, -1, 0, 1);
		cv::Sobel(eyeImage_norm, imFilt_down, -1, 0, 1);
		cv::Sobel(1 - eyeImage_norm, imFilt_left, -1, 1, 0);
		cv::Sobel(eyeImage_norm, imFilt_right, -1, 1, 0);

		cv::normalize(imFilt_up, imFilt_up, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_down, imFilt_down, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_left, imFilt_left, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_right, imFilt_right, 0, 1, cv::NORM_MINMAX, -1);

		float beta = pupil_beta;// 20;
		int n = 1;
		int post_val_prev = 0;
		double postDiff_thold = 0;
		double postDiff_thold_weight = 0.1;
		int maxN = 10;
		int x1, x2, y1, y2;
		float delta = 15;

		cv::Mat log_likelihood;  //KL are these big?
		double mean_log_MAP_prev = 0;
		//cv::RotatedRect ellipse;
		cv::Mat imFilt_crop;

		while (1) {
			//cout << n << endl;
			cv::Mat eyeImage_clone2 = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone(); //eyeImage_clone.clone();

			if (0) { // PLOT
				for (int i = 0; i < pupil_edge.size(); i++) {
					circle(eyeImage_clone2, pupil_edge[i], 2, 150, -1, 8);
					imshow("kuva", eyeImage_clone2);
				}
				cv::waitKey(0);
			}

			//ellipse = cv::fitEllipse(pupil_edge);
			pupilEllipse = cv::fitEllipse(pupil_edge);
			pupil_edge.clear();

			//float a = ellipse.size.height / 2;
			//float b = ellipse.size.width / 2;
			float a = pupilEllipse.size.height / 2;
			float b = pupilEllipse.size.width / 2;
			float phi = (270 - pupilEllipse.angle) *CV_PI / 180;
			float Xc = pupilEllipse.center.x;
			float Yc = pupilEllipse.center.y;
			std::vector<cv::Point2f> ellipse_points;
			double angle_step = 0.3; // 2*CV_PI / angle_step > 5 must hold !
			for (float t = 0; t < 2 * CV_PI; t = t + angle_step) {
				float ellipse_point_X = std::min(std::max(float(0), Xc + a*std::cos(t)*std::cos(phi) - b*std::sin(t)*std::sin(phi)), float(x_max - x_min)); // float(eyeImage_cropped.cols)); KL - this should be the same number, eh?
				float ellipse_point_Y = std::min(std::max(float(0), Yc + a*std::cos(t)*std::sin(phi) + b*std::sin(t)*std::cos(phi)), float(y_max - y_min)); // float(eyeImage_cropped.rows));
				ellipse_points.push_back(cv::Point2f(ellipse_point_X, ellipse_point_Y));
				//ellipse_points.push_back(cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)));
				// circle(eyeImage_clone2,  cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)), 2, 150, -1, 8);
			}
			// imshow("kuva", eyeImage_clone2);
			// cv::waitKey(1);

			std::vector<double> log_MAP;
			for (int i = 0; i < ellipse_points.size(); i++) {

				x1 = std::max(float(0), ellipse_points[i].x - delta);
				x2 = std::min(float(x_max - x_min/*eyeImage_cropped.cols*/), ellipse_points[i].x + delta);
				y1 = std::max(float(0), ellipse_points[i].y - delta);
				y2 = std::min(float(y_max - y_min/*eyeImage_cropped.rows*/), ellipse_points[i].y + delta);

				if (Yc - ellipse_points[i].y >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the upper sector of the pupil
					imFilt_crop = imFilt_up(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the eye lid often produces a disturbing edge --> divide beta by half for the upper sector of the pupil
				}
				if (ellipse_points[i].y - Yc >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the bottom sector of the pupil
					imFilt_crop = imFilt_down(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the lower eye lid often produces a disturbing edge --> scale beta by 0.75 for the lower sector of the pupil
				}
				if (Xc - ellipse_points[i].x >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the left sector of the pupil
					imFilt_crop = imFilt_left(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}
				if (ellipse_points[i].x - Xc >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the right sector of the pupil
					imFilt_crop = imFilt_right(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}

				log_likelihood = beta * imFilt_crop;

				// Form the coordinates for which to evaluate the prior
				// perhaps Eigen would be quicker here?
				cv::Mat xv(1, x2 - x1, CV_32F);
				cv::Mat yv(y2 - y1, 1, CV_32F);
				int ii = 0; for (int x = x1; x < x2; x++) { xv.at<float>(0, ii++) = x; }
				ii = 0;     for (int y = y1; y < y2; y++) { yv.at<float>(ii++, 0) = y; }
				cv::Mat xx = repeat(xv, yv.total(), 1);
				cv::Mat yy = repeat(yv, 1, xv.total());
				cv::Mat xcoords = xx.reshape(0, 1);
				cv::Mat ycoords = yy.reshape(0, 1);

				cv::Mat dist_x2, dist_y2;
				cv::pow(ellipse_points[i].x - xcoords, 2, dist_x2);
				cv::pow(ellipse_points[i].y - ycoords, 2, dist_y2);
				cv::Mat dist_vec;
				cv::sqrt(dist_x2 + dist_y2, dist_vec);
				cv::Mat dist = dist_vec.reshape(0, yv.rows);
				cv::Mat log_prior = -dist;
				cv::Mat log_post = log_likelihood + log_prior;

				minMaxLoc(log_post, &min_val, &max_val, &min_loc, &max_loc);

				max_loc.x = max_loc.x + x1;
				max_loc.y = max_loc.y + y1;

				pupil_edge.push_back(max_loc);
				log_MAP.push_back(max_val);
			}

			if (n == 2) {
				postDiff_thold = std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) * postDiff_thold_weight;
			}  // get the threshold automatically

			//while gets repeated until 
			if (n++ > maxN || (std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) < postDiff_thold)) {
				break;
			}

			mean_log_MAP_prev = cv::mean(log_MAP)[0];
			log_MAP.clear();

		}  // end of while(1)


		if (0) {  // PLOT
			for (int i = 0; i < pupil_edge.size(); i++) {
				circle(eyeImage_clone, pupil_edge[i], 2, 150, -1, 8);
				imshow("Lopullinen kuva", eyeImage_clone);
			}
			cv::waitKey(1);
		}

		pupilEllipse = cv::fitEllipse(pupil_edge);
		pupilEllipse.center.x = pupilEllipse.center.x + x_min;
		pupilEllipse.center.y = pupilEllipse.center.y + y_min;
		pupilEllipse.angle = (270 - pupilEllipse.angle) *CV_PI / 180;

		//return ellipse;

	}
}


void PupilEstimator::getPupilEllipseThreaded(cv::RotatedRect &pupilEllipse, cv::Mat eyeImage_opened, cv::Point2d pupil_center, cv::Mat pupil_kernel, cv::Mat pupil_element)
{

  // Onko t‰m‰ funktio turha? t: Miika


	cv::Mat eyeImage_filtered;
	//cv::Mat eyeImage_cropped; can be removed(?)
	cv::Mat eyeImage_closed;
	cv::Mat eyeImage_binary;
	cv::Mat eyeImage_binary_8bit;
	cv::Mat eyeImage_sum;
	//cv::Mat eyeImage_sum_float;
	cv::Mat eyeImage_sum_scaled;
	cv::Mat connComps;
	cv::Mat connComps_32F;
	cv::Mat pupil_blob;
	cv::Mat pupil_blob_8bit;
	cv::Mat connComps_aux;


	float Svert = eyeImage_opened.rows;
	float Shori = eyeImage_opened.cols;
	cv::Point2d pupil_center_in = pupil_center;

	// Crop the image around the pupil center
	x_min = std::max(float(0), float(pupil_center.x) - delta_x);
	x_max = std::min(Shori, float(pupil_center.x) + delta_x);
	y_min = std::max(float(0), float(pupil_center.y) - delta_y);
	y_max = std::min(Svert, float(pupil_center.y) + delta_y);
	//eyeImage_cropped = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)); //t‰‰ on laillisuuden rajamailla, funktiokutsun mukaan ton ei pit‰is muokkaantua, 
	//mutta nyt croppedin modaaminen muokkaisi sit‰. T‰‰ tosin vaan filtterˆid‰‰n, joten koko cropped on turha?
	pupil_center.x = pupil_center.x - x_min;   // Pupil center in the cropped image
	pupil_center.y = pupil_center.y - y_min;   // Pupil center in the cropped image

	// Filter the image (and crop away edges)
	cv::filter2D(eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)), // <- was eyeImage_cropped
		eyeImage_filtered, -1, pupil_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

	int delta_edge = 10;  // Crop the edges away
	if (delta_edge > delta_x || delta_edge > delta_y) { std::cout << "delta_edge is larger than the image half " << std::endl; exit(-1); }

	//cv::Mat eyeImage_filtered_cropped = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge));
	morphologyEx(eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)),
		eyeImage_closed, cv::MORPH_CLOSE, pupil_element);
	eyeImage_sum = eyeImage_filtered(cv::Range(delta_edge, eyeImage_filtered.rows - delta_edge), cv::Range(delta_edge, eyeImage_filtered.cols - delta_edge)) + eyeImage_closed; //cv::add might use optimized code?
	pupil_center.x = pupil_center.x - delta_edge;   // Pupil center (x) in the cropped image
	pupil_center.y = pupil_center.y - delta_edge;   // Pupil center (y) in the cropped image

	// Scale the image
	//avoid conversion operations by doing it in 8-bit space?
	/*
	eyeImage_sum.convertTo(eyeImage_sum_float, CV_32F);
	cv::normalize(eyeImage_sum_float, eyeImage_sum_scaled, 0, 1, cv::NORM_MINMAX, -1);

	// Binarize the image
	float thold = 0.15;  // was 0.15
	threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 1, cv::THRESH_BINARY_INV);

	// Get the connected components of the binary image
	eyeImage_binary.convertTo(eyeImage_binary_8bit, CV_8UC1);
	*/
	int thold = floor(0.15 * 255);
	cv::normalize(eyeImage_sum, eyeImage_sum_scaled, 0, 255, cv::NORM_MINMAX, -1);
	cv::threshold(eyeImage_sum_scaled, eyeImage_binary, thold, 255, cv::THRESH_BINARY_INV);
	//the replacement for the commented bit above ends here
	//cv::connectedComponents(eyeImage_binary_8bit, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)
	cv::connectedComponents(eyeImage_binary, connComps, 4, CV_32S);  // == "bwlabel"  (in opencv3 only)

	// Get the pupil blob, being the area which encloses the pupil center
	int pupilLabel = connComps.at<int>(pupil_center.y, pupil_center.x);  // index of the pupil area
	connComps.convertTo(connComps_32F, CV_32F);
	//KL <- muutettu pysyv‰ksi cv::Mat connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	connComps_aux = cv::abs(connComps_32F - pupilLabel);  // here the pupil area is all zeros, other areas have larger indices
	threshold(connComps_aux, pupil_blob, 0.5, 1, cv::THRESH_BINARY_INV);  // get the pupil area

	// Find the contours of the pupil blob
	pupil_blob.convertTo(pupil_blob_8bit, CV_8U);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(pupil_blob_8bit, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	//select the longest contour
	pupil_contour = contours[0];
	for (int i = 1; i<contours.size(); i++) {
		if (contours[i].size() > pupil_contour.size()) {
			pupil_contour = contours[i];
		}
	}

	//not a very neat solution?
	while (pupil_contour.size() < 6) {  // If pupil_contour happens to contain less than 5 points (this shouldn't normally happen), add the first point until N>5
		pupil_contour.push_back(pupil_contour[0]);
	}

	if (0) { // Draw the found contours
		cv::Mat drawing = cv::Mat::zeros(connComps.size(), CV_8UC3);
		cv::drawContours(drawing, contours, -1, 250, 2, 8);
		//cv::drawContours( drawing, pupil_contour, -1, 250, 2, 8);
		namedWindow("Contours", cv::WINDOW_AUTOSIZE); imshow("Contours", drawing); cv::waitKey(1);
	}

	if (!ITERATE) {
		// Compute the pupil_contour in the original (incoming) image coordinates
		for (int i = 0; i<pupil_contour.size(); i++) {
			pupil_contour[i].x = pupil_contour[i].x + x_min + delta_edge;
			pupil_contour[i].y = pupil_contour[i].y + y_min + delta_edge;
		}
		//cv::RotatedRect pupilEllipse = cv::fitEllipse(pupil_contour);
		pupilEllipse = cv::fitEllipse(pupil_contour);

		//return pupilEllipse;
	}

	else {

		//  %%%%%%%%%%%%%%%%%%%%%  FINAL ITERATION  %%%%%%%%%%%%%%%%%%%%%%%%

		// 	eyeImage_clone = eyeImage_cropped.clone();
		eyeImage_clone = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone();

		//removed the need for cropped -> use the clone
		//eyeImage_cropped.convertTo(eyeImage_cropped, CV_32F);
		//cv::normalize(eyeImage_cropped, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);
		eyeImage_clone.convertTo(eyeImage_clone, CV_32F);
		cv::normalize(eyeImage_clone, eyeImage_norm, 0, 1, cv::NORM_MINMAX, -1);

		// Compute the pupil_contour in the cropped image coordinates
		for (int i = 0; i < pupil_contour.size(); i++) {
			pupil_edge.push_back(cv::Point2d(pupil_contour[i].x + delta_edge, pupil_contour[i].y + delta_edge));
			// circle(eyeImage_clone, cv::Point2d(pupil_contour[i].x + delta_edge , pupil_contour[i].y + delta_edge) , 2, 250, -1, 8);
		}

		cv::Mat edge_filt = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, 1, -1);

		cv::Sobel(1 - eyeImage_norm, imFilt_up, -1, 0, 1);
		cv::Sobel(eyeImage_norm, imFilt_down, -1, 0, 1);
		cv::Sobel(1 - eyeImage_norm, imFilt_left, -1, 1, 0);
		cv::Sobel(eyeImage_norm, imFilt_right, -1, 1, 0);

		cv::normalize(imFilt_up, imFilt_up, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_down, imFilt_down, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_left, imFilt_left, 0, 1, cv::NORM_MINMAX, -1);
		cv::normalize(imFilt_right, imFilt_right, 0, 1, cv::NORM_MINMAX, -1);

		float beta = 20;
		int n = 1;
		int post_val_prev = 0;
		double postDiff_thold = 0;
		double postDiff_thold_weight = 0.1;
		int maxN = 10;
		int x1, x2, y1, y2;
		float delta = 15;

		cv::Mat log_likelihood;  //KL are these big?
		double mean_log_MAP_prev = 0;
		cv::RotatedRect ellipse;
		cv::Mat imFilt_crop;

		while (1) {
			//cout << n << endl;
			cv::Mat eyeImage_clone2 = eyeImage_opened(cv::Range(y_min, y_max), cv::Range(x_min, x_max)).clone(); //eyeImage_clone.clone();

			if (0) { // PLOT
				for (int i = 0; i < pupil_edge.size(); i++) {
					circle(eyeImage_clone2, pupil_edge[i], 2, 150, -1, 8);
					imshow("kuva", eyeImage_clone2);
				}
				cv::waitKey(0);
			}
			
			ellipse = cv::fitEllipse(pupil_edge);
			pupil_edge.clear();

			float a = ellipse.size.height / 2;
			float b = ellipse.size.width / 2;
			float phi = (270 - ellipse.angle) *CV_PI / 180;
			float Xc = ellipse.center.x;
			float Yc = ellipse.center.y;
			std::vector<cv::Point2f> ellipse_points;
			double angle_step = 0.3; // 2*CV_PI / angle_step > 5 must hold !
			for (float t = 0; t < 2 * CV_PI; t = t + angle_step) {
			  float ellipse_point_X = std::min(std::max(float(0), Xc + a*std::cos(t)*std::cos(phi) - b*std::sin(t)*std::sin(phi)), float(x_max - x_min)); // (x_max - x-min == float(eyeImage_cropped.cols) )
			  float ellipse_point_Y = std::min(std::max(float(0), Yc + a*std::cos(t)*std::sin(phi) + b*std::sin(t)*std::cos(phi)), float(y_max - y_min)); // (y_max - y_min == float(eyeImage_cropped.rows) )
				ellipse_points.push_back(cv::Point2f(ellipse_point_X, ellipse_point_Y));
				//ellipse_points.push_back(cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)));
				// circle(eyeImage_clone2,  cv::Point2f(Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi) , Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)), 2, 150, -1, 8);
			}
			// imshow("kuva", eyeImage_clone2);
			// cv::waitKey(1);

			std::vector<double> log_MAP;
			for (int i = 0; i < ellipse_points.size(); i++) {

				x1 = std::max(float(0), ellipse_points[i].x - delta);
				x2 = std::min(float(x_max - x_min/*eyeImage_cropped.cols*/), ellipse_points[i].x + delta);
				y1 = std::max(float(0), ellipse_points[i].y - delta);
				y2 = std::min(float(y_max - y_min/*eyeImage_cropped.rows*/), ellipse_points[i].y + delta);

				if (Yc - ellipse_points[i].y >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the upper sector of the pupil
					imFilt_crop = imFilt_up(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the eye lid often produces a disturbing edge --> divide beta by half for the upper sector of the pupil
				}
				if (ellipse_points[i].y - Yc >= abs(Xc - ellipse_points[i].x)) {  // Form likelihood for the bottom sector of the pupil
					imFilt_crop = imFilt_down(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
					imFilt_crop = imFilt_crop * 0.5;  // Heuristics: the lower eye lid often produces a disturbing edge --> scale beta by 0.75 for the lower sector of the pupil
				}
				if (Xc - ellipse_points[i].x >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the left sector of the pupil
					imFilt_crop = imFilt_left(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}
				if (ellipse_points[i].x - Xc >= abs(Yc - ellipse_points[i].y)) {  // Form likelihood for the right sector of the pupil
					imFilt_crop = imFilt_right(cv::Range(float(y1), float(y2)), cv::Range(float(x1), float(x2)));
				}

				log_likelihood = beta * imFilt_crop;

				// Form the coordinates for which to evaluate the prior
				// perhaps Eigen would be quicker here?
				cv::Mat xv(1, x2 - x1, CV_32F);
				cv::Mat yv(y2 - y1, 1, CV_32F);
				int ii = 0; for (int x = x1; x < x2; x++) { xv.at<float>(0, ii++) = x; }
				ii = 0;     for (int y = y1; y < y2; y++) { yv.at<float>(ii++, 0) = y; }
				cv::Mat xx = repeat(xv, yv.total(), 1);
				cv::Mat yy = repeat(yv, 1, xv.total());
				cv::Mat xcoords = xx.reshape(0, 1);
				cv::Mat ycoords = yy.reshape(0, 1);

				cv::Mat dist_x2, dist_y2;
				cv::pow(ellipse_points[i].x - xcoords, 2, dist_x2);
				cv::pow(ellipse_points[i].y - ycoords, 2, dist_y2);
				cv::Mat dist_vec;
				cv::sqrt(dist_x2 + dist_y2, dist_vec);
				cv::Mat dist = dist_vec.reshape(0, yv.rows);
				cv::Mat log_prior = -dist;
				cv::Mat log_post = log_likelihood + log_prior;

				minMaxLoc(log_post, &min_val, &max_val, &min_loc, &max_loc);

				max_loc.x = max_loc.x + x1;
				max_loc.y = max_loc.y + y1;

				pupil_edge.push_back(max_loc);
				log_MAP.push_back(max_val);
			}

			if (n == 2) {
				postDiff_thold = std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) * postDiff_thold_weight;
			}  // get the threshold automatically

			//while gets repeated until 
			if (n++ > maxN || (std::abs(cv::mean(log_MAP)[0] - mean_log_MAP_prev) < postDiff_thold)) {
				break;
			}

			mean_log_MAP_prev = cv::mean(log_MAP)[0];
			log_MAP.clear();

		}  // end of while(1)


		if (0) {  // PLOT
			for (int i = 0; i < pupil_edge.size(); i++) {
				circle(eyeImage_clone, pupil_edge[i], 2, 150, -1, 8);
				imshow("Lopullinen kuva", eyeImage_clone);
			}
			cv::waitKey(1);
		}

		ellipse = cv::fitEllipse(pupil_edge);
		ellipse.center.x = ellipse.center.x + x_min;
		ellipse.center.y = ellipse.center.y + y_min;
		ellipse.angle = (270 - ellipse.angle) *CV_PI / 180;

		pupilEllipse = ellipse;

		//return ellipse;
	}
}
