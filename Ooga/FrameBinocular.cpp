#include "FrameBinocular.h"

TBinocularFrame::TBinocularFrame()
{
	// init images
	img_eye_L = new cv::UMat();
	img_eye_R = new cv::UMat();
	img_scene = new cv::UMat();

	//timestamp = -1;
	int64 number = -1;

	//refcount++;
}


TBinocularFrame::~TBinocularFrame()
{
	try{
		if (img_eye_L != nullptr){
			delete img_eye_L;
		}
		if (img_eye_R != nullptr){
			delete img_eye_R;
		}
		if (img_scene != nullptr){
			delete img_scene;
		}

		//remove aux images
		while (aux_images.size() > 0){
			delete aux_images[aux_images.size() - 1];
			aux_images.pop_back();
		}

	}
	catch (int e) {
		std::cerr << "it was frame del: " << e << std::endl;
	}
}

//this deep copies the given image, not used currently
void TBinocularFrame::setImg(FrameSrc f, cv::UMat *img)
{
	if (f == FrameSrc::EYE_L){
		//img.copyTo(this->img_eye);
		img->copyTo(*(this->img_eye_L));
		//this->img_eye_L = img;
	}
	else if (f == FrameSrc::EYE_R){
		//img.copyTo(this->img_eye);
		img->copyTo(*(this->img_eye_R));
		//this->img_eye_R = img;
	}
	else if (f == FrameSrc::SCENE){
		//img.copyTo(this->img_scene);
		img->copyTo(*(this->img_scene));
		//this->img_scene = img;
	}
}

cv::UMat *TBinocularFrame::getImg(FrameSrc f)
{
	if (f == FrameSrc::EYE_L){
		return this->img_eye_L;
	}
	if (f == FrameSrc::EYE_R){
		return this->img_eye_R;
	}
	else if (f == FrameSrc::SCENE){
		return this->img_scene;
	}

}

void TBinocularFrame::pushAuxImg(cv::UMat *img){
	aux_images.push_back(img);
}


bool TBinocularFrame::popAuxImg(cv::UMat *&img){
	if (!aux_images.empty()){
		img = aux_images.back();
		aux_images.pop_back();
		return true;
	}
	else{
		//img = nullptr; TODO: PEEK FUNCTION?!
		return false;
	}
}

