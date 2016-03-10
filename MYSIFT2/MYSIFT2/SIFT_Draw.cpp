#include "StdAfx.h"
#include "SIFT_Draw.h"


CSift_Draw::CSift_Draw(Mat _img):img(_img)
{
}


CSift_Draw::~CSift_Draw(void)
{
}

void CSift_Draw::draw_oxfd_feature( struct feature* feat, CvScalar color )
{
	double m[4] = { feat->a, feat->b, feat->b, feat->c };
	double v[4] = { 0 };
	double e[2] = { 0 };
	CvMat M, V, E;
	double alpha, l1, l2;

	/* compute axes and orientation of ellipse surrounding affine region */
	cvInitMatHeader( &M, 2, 2, CV_64FC1, m, CV_AUTOSTEP );
	cvInitMatHeader( &V, 2, 2, CV_64FC1, v, CV_AUTOSTEP );
	cvInitMatHeader( &E, 2, 1, CV_64FC1, e, CV_AUTOSTEP );
	cvEigenVV( &M, &V, &E, DBL_EPSILON ,-1,-1);
	l1 = 1 / sqrt( e[1] );
	l2 = 1 / sqrt( e[0] );
	alpha = -atan2( v[1], v[0] );
	alpha *= 180 / CV_PI;

	ellipse(img,Point(feat->x, feat->y ),Size(12,11),alpha,0,360, CV_RGB(0,0,0), 3, 8, 0);
	ellipse(img,Point(feat->x, feat->y ),Size(12,11),alpha,0,360, color, 1, 8, 0);
	line(img,Point(feat->x+2, feat->y ),Point(feat->x-2,feat->y),color, 1, 8, 0);
	line(img,Point(feat->x, feat->y+2 ),Point(feat->x,feat->y-2), color, 1, 8, 0);
}
void CSift_Draw::draw_oxfd_features(struct feature* feat, int n )
{
	CvScalar color = CV_RGB( 255, 255, 255 );
	int i;

	if( img.channels() > 1 )
		color = CV_RGB(255,255,0);
	for( i = 0; i < n; i++ )
		draw_oxfd_feature(  feat + i, color );
}
void CSift_Draw::draw_lowe_feature(  struct feature* feat, CvScalar color )
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound( feat->x );
	start_y = cvRound( feat->y );
	scl = feat->scl;
	ori = feat->ori;
	len = cvRound( scl * scale );
	hlen = cvRound( scl * hscale );
	blen = len - hlen;
	end_x = cvRound( len *  cos( ori ) ) + start_x;
	end_y = cvRound( len * -sin( ori ) ) + start_y;
	h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
	h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
	h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
	h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
	start = cvPoint( start_x, start_y );
	end = cvPoint( end_x, end_y );


	h1 = cvPoint( h1_x, h1_y );
	h2 = cvPoint( h2_x, h2_y );

	line(img,start,end, color, 1, 8, 0);

	line(img,end,h1, color, 1, 8, 0);
	line(img,end,h2, color, 1, 8, 0);
}
void CSift_Draw::draw_lowe_features( struct feature* feat, int n )
{
	CvScalar color = CV_RGB( 255, 255, 255 );
	int i;

	if( img.channels() > 1 )
		color =CV_RGB(255,0,255);
	for( i = 0; i < n; i++ )
		draw_lowe_feature( feat + i, color );
}
void CSift_Draw::draw_features( struct feature* feat, int n )
{
	int type;
	type = feat[0].type;
	switch( type )
	{
	case 0:
		draw_oxfd_features(  feat, n );
		break;
	case 1:
		draw_lowe_features(  feat, n );
		break;
	default:
		fprintf( stderr, "Warning: draw_features(): unrecognized feature" \
			" type, %s, line %d\n", __FILE__, __LINE__ );
		break;
	}
}
void CSift_Draw::show(char *windowname){
	imshow(windowname,img);
	imwrite("match.jpg",img);
	waitKey(0);
}