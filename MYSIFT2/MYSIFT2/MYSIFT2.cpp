// MYSIFT2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "SIFT_Features.h"
#include "SIFT_KDTree.h"
#include "SIFT_Matches.h"
#include "SIFT_Draw.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{

	    CSift_Feathures csf1,csf2;

		Mat img1=imread("..\\box.png");  
		csf1._sift_features(img1);
		Mat img2=imread("..\\box_scene.png");  
		csf2._sift_features(img2);

		Mat feat_img1,feat_img2;
		img1.copyTo(feat_img1);
		img2.copyTo(feat_img2);

		CSift_Draw csd(feat_img1);
		csd.draw_features(csf1.features,csf1.n);
		csd.show("sift1");

		CSift_Draw csd2(feat_img2);
		csd2.draw_features(csf2.features,csf2.n);
		csd2.show("sift2");


		CSift_Matchs csm;
		Mat stacked=csm.stack_imgs(img1,img2);
		csm.match(csf1.features,csf1.n,csf2.features,csf2.n,img1,stacked);
		
		imshow("aaa",stacked);
		waitKey(0);

	return 0;
}

