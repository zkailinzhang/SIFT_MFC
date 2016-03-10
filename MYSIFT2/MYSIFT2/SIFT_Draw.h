#pragma once



#include "SIFT_Features.h"
#include <opencv2\opencv.hpp>
using namespace cv;

class CSift_Draw
{
public:
	CSift_Draw(Mat _img);
	~CSift_Draw(void);
	void draw_oxfd_feature(struct feature* feat, CvScalar color );
	void draw_oxfd_features(  struct feature* feat, int n );
	void draw_lowe_feature( struct feature* feat, CvScalar color );
	void draw_lowe_features(  struct feature* feat, int n );
	void draw_features( struct feature* feat, int n );
	void show(char *windowname);

public:
	Mat img;
};

