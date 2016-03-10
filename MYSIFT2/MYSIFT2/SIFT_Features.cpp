#include "StdAfx.h"
#include "SIFT_Features.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstring>
#include <cmath>

using namespace cv;
using namespace std;

int aa=0,bb=0,cc=0,dd=0,ee=0;

CSift_Feathures::CSift_Feathures(void):intvls(3),contr_thr(0.04),curv_thr(10),img_dbl(1),sigma(1.6),
	descr_width(4),descr_hist_bins(8)
{
}

CSift_Feathures::~CSift_Feathures(void)
{
	cvReleaseMemStorage( &storage );
	imgsrc.release();
	release_pyr( gauss_pyr, octvs, intvls + 3 );
	release_pyr( dog_pyr, octvs, intvls + 2 );
}


Mat CSift_Feathures::convert_to_gray32(Mat &img ){
	Mat gray8,gray32;
	//int r,c;
	gray8.create(Size(img.cols,img.rows),CV_8UC1);
	gray32.create(Size(img.cols,img.rows),CV_32FC1);
	if (img.channels()==1)
		gray8=img.clone();
	else
		cvtColor(img,gray8,CV_RGB2GRAY);

	
	//gray8.convertTo(gray32,CV_32FC1,1.0/255.0,0);
	cvConvertScale(&(IplImage)gray8,&(IplImage)gray32,1.0/255.0,0);
	gray8.release();
	return gray32;
}

Mat CSift_Feathures::create_init_img(Mat img){
	Mat gray_img;
	float sig_diff;
	gray_img=convert_to_gray32(img);
	if (img_dbl)
	{
		sig_diff=sqrt(sigma*sigma-0.5*0.5*4);
		imgsrc.create(Size(img.cols*2,img.rows*2),CV_32FC1);//构造第0层
		cvResize(&(IplImage)gray_img,&(IplImage)imgsrc,CV_INTER_CUBIC);
		cvSmooth(&(IplImage)imgsrc,&(IplImage)imgsrc,CV_GAUSSIAN,0,0,sig_diff,sig_diff);
		gray_img.release();
		return imgsrc;
	}
	else
	{
		sig_diff=sqrt(sigma*sigma-0.5*0.5);
		cvSmooth(&(IplImage)gray_img,&(IplImage)gray_img,CV_GAUSSIAN,0,0,sig_diff,sig_diff);
		return gray_img;
	}
}

Mat CSift_Feathures::downsample(Mat img )
{
	Mat smaller;
	smaller.create(Size(img.cols/2,img.rows/2),img.type());
	cvResize( &(IplImage)img, &(IplImage)smaller, CV_INTER_NN );
	return smaller;
}

void CSift_Feathures::build_gauss_pyr(){
	double *sig=(double *)calloc(intvls+3,sizeof(double));
	double sig_total,sig_prev,k;
	int i,o;
	gauss_pyr.resize(octvs);
	for (i=0;i<octvs;i++)
		gauss_pyr[i].resize(intvls+3);
	sig[0] = sigma;
	k = pow( 2.0, 1.0 / intvls );
	for( i = 1; i < intvls + 3; i++ )
	{
		sig_prev = pow( k, i - 1 ) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
	}

	for( o = 0; o < octvs; o++ )
	{	for( i = 0; i < intvls + 3; i++ )
		{
			if( o == 0  &&  i == 0 )
				gauss_pyr[o][i]=imgsrc.clone();
			else if( i == 0 )
				gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );
			else
			{
				gauss_pyr[o][i].create(Size(gauss_pyr[o][i-1].cols,gauss_pyr[o][i-1].rows),CV_32FC1);
				//GaussianBlur(gauss_pyr[o][i-1],gauss_pyr[o][i],Size(3,3),sig[i], sig[i]);

				cvSmooth( &(IplImage)gauss_pyr[o][i-1], &(IplImage)gauss_pyr[o][i],CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
			}
		}

	}

// 	stringstream sss;
// 	int kkkk = 0;
// 
// 	cout<<endl<<"   高斯尺度空间图片数目： "<< octvs *(intvls + 3)<<endl;
// 	for( int oo = 0; oo < octvs; oo++ )
// 	{	for(int ii = 0; ii < intvls + 3; ii++ )
// 	{
// 
// 		sss<< "gaussian_classroom_"<< kkkk<<".jpg";
// 		Mat img2( gauss_pyr[oo][ii]);                     //true 深拷贝 全黑；   0 前拷贝 ok显示   
// 		Mat img5;
// 		img2.convertTo(img5,CV_8UC1,255,0);
// 		//imshow("x",img5);
// 		imwrite(sss.str(),img5);
// 		//waitKey(0);
// 
// 		kkkk++;
// 		sss.str("");
// 	}}



		free( sig );
}

void CSift_Feathures::build_dog_pyr()
{
	int i, o;
	dog_pyr.resize(octvs);
	for( i = 0; i < octvs; i++ )
		dog_pyr[i].resize(intvls+2);
	for( o = 0; o < octvs; o++ )
	{	for( i = 0; i < intvls + 2; i++ )
		{
			dog_pyr[o][i].create(Size(gauss_pyr[o][i].cols,gauss_pyr[o][i].rows),CV_32FC1);
			//cvAbsDiff(&(IplImage)gauss_pyr[o][i+1], &(IplImage)gauss_pyr[o][i], &(IplImage)dog_pyr[o][i]);
			//cvSub(&(IplImage)gauss_pyr[o][i+1], &(IplImage)gauss_pyr[o][i], &(IplImage)dog_pyr[o][i],NULL);

			absdiff(gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i]);
		}
	}

// 	stringstream ss;
// 	int kkk = 0;
// 
// 	for( int j = 0; j < octvs; j++ )
// 	{	for( int k = 0; k < intvls + 2; k++ )
// 	{
// 
// 
// 		ss<<"dog_classroom_"<<kkk<<".jpg";
// 		Mat img3(dog_pyr[j][k]);
// 		//imshow("x",img3);
// 		Mat img4;
// 		img3.convertTo(img4,CV_8UC1,255,0);
// 		//img4.create(1 , img3.size , img3.type());
// 		//cvtColor(img3,img4,CV_BGR2GRAY);
// 		imwrite(ss.str(),img4);
// 		//waitKey(0);
// 		kkk++;
// 		ss.str("");
// 	}
// 	}




}

float CSift_Feathures::pixval32f(Mat img, int r, int c )
{
	float value=(img.ptr<float>(r))[c];
	return value;
}

int CSift_Feathures::is_extremum( int octv, int intvl, int r, int c )
{
	float val = pixval32f( dog_pyr[octv][intvl], r, c );
	int i, j, k;
	if( val > 0 )
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )

					//与周围的26个像素作对比找最大值
					if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	else
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )

					//与周围的26个像素作对比找最小值
					if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	return 1;
}

//计算三维偏导数
CvMat* CSift_Feathures::deriv_3D(  int octv, int intvl, int r, int c )
{
	CvMat* dI;
	double dx, dy, ds;

	//实际上在离散数据中计算偏导数是通过相邻像素的相减来计算的
	//比如说计算x方向的偏导数dx，则通过该向所的x
	//方向的后一个减去前一个然后除以2即可求的dx
	dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
		pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
	dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
		pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
	ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
		pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;

	dI = cvCreateMat( 3, 1, CV_64FC1 );
	cvmSet( dI, 0, 0, dx );
	cvmSet( dI, 1, 0, dy );
	cvmSet( dI, 2, 0, ds );

	return dI;
}

//计算二次导数(三维海森矩阵)
CvMat* CSift_Feathures::hessian_3D( int octv, int intvl, int r, int c )
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	//二次导数 前面像素+后面像素-2*中间像素
	v = pixval32f( dog_pyr[octv][intvl], r, c );
	dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
		pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
	dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
		pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
	dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
		pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );

	//（（左上+右下）-（右上+左下））/4; 
	dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
		pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
		pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
		pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
	dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
		pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
		pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
		pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
	dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
		pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
		pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
		pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;

	H = cvCreateMat( 3, 3, CV_64FC1 );
	cvmSet( H, 0, 0, dxx );
	cvmSet( H, 0, 1, dxy );
	cvmSet( H, 0, 2, dxs );
	cvmSet( H, 1, 0, dxy );
	cvmSet( H, 1, 1, dyy );
	cvmSet( H, 1, 2, dys );
	cvmSet( H, 2, 0, dxs );
	cvmSet( H, 2, 1, dys );
	cvmSet( H, 2, 2, dss );

	return H;
}

//获取亚像素位置所用到的函数
void CSift_Feathures::interp_step( int octv, int intvl, int r, int c,
	double* xi, double* xr, double* xc )
{
	CvMat* dD, * H, * H_inv, X;
	double x[3] = { 0 };

	dD = deriv_3D( octv, intvl, r, c );//3*1维
	H = hessian_3D(  octv, intvl, r, c );//3*3维

	H_inv = cvCreateMat( 3, 3, CV_64FC1 );
	cvInvert( H, H_inv, CV_SVD );//svd求模拟矩阵的逆矩阵(参数分别是带求解矩阵，逆矩阵,求解方式(奇异值分解))
	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );

	//dst = alpha*op(src1)*op(src2) + beta*op(src3)
	cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );//通用矩阵乘法(参数分别是:输入矩阵1,输入矩阵2,系数,输入矩阵3,系数,输出矩阵)
	//x是3*1维的矩阵

	cvReleaseMat( &dD );
	cvReleaseMat( &H );
	cvReleaseMat( &H_inv );

	//分别保存三个方向上的偏移值
	*xi = x[2];
	*xr = x[1];
	*xc = x[0];
}

//计算插入像素的对比度
double CSift_Feathures::interp_contr( int octv, int intvl, int r,
	int c, double xi, double xr, double xc )
{
	CvMat* dD, X, T;
	double t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
	dD = deriv_3D( octv, intvl, r, c );
	cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
	cvReleaseMat( &dD );

	return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}

struct feature* CSift_Feathures::new_feature( void )
{
	struct feature* feat;
	struct detection_data* ddata;

	feat = (feature *)malloc( sizeof( struct feature ) );
	memset( feat, 0, sizeof( struct feature ) );
	ddata =(detection_data *) malloc( sizeof( struct detection_data ) );
	memset( ddata, 0, sizeof( struct detection_data ) );
	feat->feature_data = ddata;
	feat->type = 1; //1 lowe 线段  0  椭圆

	return feat;
}
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )

//获取亚像素极值点的位置
struct feature* CSift_Feathures::interp_extremum(  int octv, int intvl,
	int r, int c )
{
	struct feature* feat;
	struct detection_data* ddata;
	double xi, xr, xc, contr;//分别为亚像素的intval,row,col的偏移offset，和对比度
	int i = 0;

	while( i < 5 )
	{
		interp_step(  octv, intvl, r, c, &xi, &xr, &xc );
		if( abs( xi ) < 0.5  &&  abs( xr ) < 0.5  &&  abs( xc ) < 0.5 )//若偏移值都小于0.5停止寻找
			break;
		//否则继续寻找极值点
		c += cvRound( xc );//cvRound对一个double型的值四舍五入，返回整数
		r += cvRound( xr );
		intvl += cvRound( xi );

		if( intvl < 1  ||
			intvl > intvls  ||
			c < 5  ||
			r < 5  ||
			c >= dog_pyr[octv][0].cols - 5  ||
			r >= dog_pyr[octv][0].rows - 5 )
		{
			ee++;
			return NULL;
		}

		i++;
	}

	//确保极值点是经过最大5步找到的
	if( i >= 5 )
	{dd++;
	return NULL;}

	//获取找到的极值点的对比度 
	contr = interp_contr(  octv, intvl, r, c, xi, xr, xc );
	//判断极值点是否小于某一个阈值
	if( abs( contr ) < contr_thr / intvls )
		return NULL;
	//若小于则认为是一个极值点
	feat = new_feature();
	ddata = feat_detection_data( feat );
	feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
	feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
	ddata->r = r;
	ddata->c = c;
	ddata->octv = octv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;

	return feat;
}

//去除边缘响应
int CSift_Feathures::is_too_edge_like( Mat dog_img, int r, int c)
{
	double d, dxx, dyy, dxy, tr, det;

	//获取特征点处的Hissian矩阵
	d = pixval32f(dog_img, r, c);
	dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
	dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
	dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
		pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
	tr = dxx + dyy;//矩阵的对角线元素和
	det = dxx * dyy - dxy * dxy;//矩阵H的行列式

	if( det <= 0 )
		return 1;

	//两个特征值的比值越大，即在某一个方向的梯度值越大,而在另一个方向的梯度值越小，而边缘恰恰就是这种情况
	//所以为了剔除边缘响应点，需要让该比值小于一定的阈值
	if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
		return 0;
	return 1;
}

//1：极值点检测以及关键点定位消除边缘响应
void CSift_Feathures::scale_space_extrema(  )
{
	double prelim_contr_thr = 0.5 * contr_thr / intvls;
	struct feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;//w, h;

	featuresSeq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
	for( o = 0; o < octvs; o++ )
	{	for( i = 1; i <= intvls; i++ )
			for(r = 5; r < dog_pyr[o][0].rows-5; r++)
				for(c = 5; c < dog_pyr[o][0].cols-5; c++)
					/* perform preliminary check on contrast */
					if( abs( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
					{	if( is_extremum(  o, i, r, c ) )
						{
							aa++;
							feat = interp_extremum( o, i, r, c);
							if( feat )
							{
								ddata = feat_detection_data( feat );
								if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
									ddata->r, ddata->c) )
								{
									//不是边缘点就加入极值点
									cvSeqPush( featuresSeq, feat );
								}
								else								
								{
									free( ddata );
									cc++;
								}
									
								free( feat );
							}
						}
					}
	}

	
}


//2：计算特征向量的尺度
void CSift_Feathures::calc_feature_scales()
{
	struct feature* feat;
	struct detection_data* ddata;
	double intvl;
	int i, n;

	n = featuresSeq->total;
	for( i = 0; i < n; i++ )
	{
		feat = CV_GET_SEQ_ELEM( struct feature, featuresSeq, i );
		ddata = feat_detection_data( feat );
		intvl = ddata->intvl + ddata->subintvl;
		feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
		ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
	}
}

//3:调整图像特征坐标,尺度,点的坐标大小为原来的一半
void CSift_Feathures::adjust_for_img_dbl( )
{
	struct feature* feat;
	int i, n;

	n = featuresSeq->total;
	for( i = 0; i < n; i++ )
	{
		feat = CV_GET_SEQ_ELEM( struct feature, featuresSeq, i );
		feat->x /= 2.0;
		feat->y /= 2.0;
		feat->scl /= 2.0;
		feat->img_pt.x /= 2.0;
		feat->img_pt.y /= 2.0;
	}
}

//4:计算所给像素的梯度大小和方向
int CSift_Feathures::calc_grad_mag_ori( Mat img, int r, int c, double* mag, double* ori )
{
	double dx, dy;

	if( r > 0  &&  r < img.rows - 1  &&  c > 0  &&  c < img.cols - 1 )
	{
		dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
		dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
		*mag = sqrt( dx*dx + dy*dy );//梯度大小
		*ori = atan2( dy, dx );//梯度方向
		return 1;
	}

	else
		return 0;
}

double* CSift_Feathures::ori_hist( Mat img, int r, int c, int n, int rad, double sigma)
{
	double* hist;
	double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	hist =(double *) calloc( n, sizeof( double ) );
	exp_denom = 2.0 * sigma * sigma;
	for( i = -rad; i <= rad; i++ )
		for( j = -rad; j <= rad; j++ )
			if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
			{
				w = exp( -( i*i + j*j ) / exp_denom );
				bin = cvRound( n * ( ori + CV_PI ) / PI2 );
				bin = ( bin < n )? bin : 0;
				hist[bin] += w * mag;
			}

			return hist;
}

void CSift_Feathures::smooth_ori_hist( double* hist, int n )
{
	double prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n-1];
	for( i = 0; i < n; i++ )
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] + 
			0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
		prev = tmp;
	}
}

double CSift_Feathures::dominant_ori( double* hist, int n )
{
	double omax;
	int maxbin, i;

	omax = hist[0];
	maxbin = 0;
	for( i = 1; i < n; i++ )
		if( hist[i] > omax )
		{
			omax = hist[i];
			maxbin = i;
		}
		return omax;
}

struct feature* CSift_Feathures::clone_feature( struct feature* feat )
{
	struct feature* new_feat;
	struct detection_data* ddata;

	new_feat = new_feature();
	ddata = feat_detection_data( new_feat );
	memcpy( new_feat, feat, sizeof( struct feature ) );
	memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
	new_feat->feature_data = ddata;

	return new_feat;
}

void CSift_Feathures::add_good_ori_features( double* hist, int n,
	double mag_thr, struct feature* feat )
{
	struct feature* new_feat;
	double bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	for( i = 0; i < n; i++ )
	{
		l = ( i == 0 )? n - 1 : i-1;
		r = ( i + 1 ) % n;

		if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
		{
			bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
			bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
			//把该关键点复制成多份关键点，并将方向值分别赋给这些复制后的关键点，
			//并且，离散的梯度方向直方图要进行插值拟合处理，来求得更精确的方向角度值
			new_feat = clone_feature( feat );
			new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;
			cvSeqPush( featuresSeq, new_feat );
			free( new_feat );
		}
	}
}


void CSift_Feathures::interp_hist_entry(  double rbin, double cbin,
	double obin, double mag )
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, * h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor( rbin );
	c0 = cvFloor( cbin );
	o0 = cvFloor( obin );
	d_r = rbin - r0;
	d_c = cbin - c0;
	d_o = obin - o0;


	for( r = 0; r <= 1; r++ )
	{
		rb = r0 + r;
		if( rb >= 0  &&  rb < descr_width )
		{
			v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
			row = hist[rb];
			for( c = 0; c <= 1; c++ )
			{
				cb = c0 + c;
				if( cb >= 0  &&  cb < descr_width )
				{
					v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
					h = row[cb];
					for( o = 0; o <= 1; o++ )
					{
						ob = ( o0 + o ) % descr_hist_bins;
						v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
						h[ob] += v_o;
					}
				}
			}
		}
	}
}

void CSift_Feathures::descr_hist(Mat img, int r, int c, double ori,
	double scl)
{
	double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
		grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
	int radius, i, j;

	hist = (double ***)calloc( descr_width, sizeof( double** ) );
	for( i = 0; i < descr_width; i++ )
	{
		hist[i] = (double **)calloc( descr_width, sizeof( double* ) );
		for( j = 0; j < descr_width; j++ )
			hist[i][j] = (double *)calloc( descr_hist_bins, sizeof( double ) );
	}

	cos_t = cos( ori );
	sin_t = sin( ori );
	bins_per_rad = descr_hist_bins / PI2;     //n==8;...
	exp_denom = descr_width * descr_width* 0.5;
	hist_width = 3.0 * scl;

	//每个子区域半径
	radius = hist_width * sqrt(2.0) * ( descr_width + 1.0 ) * 0.5 + 0.5;
	for( i = -radius; i <= radius; i++ )
		for( j = -radius; j <= radius; j++ )
		{
			//将坐标移至关键点的主方向,确保旋转不变性

			c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			rbin = r_rot + descr_width / 2 - 0.5;
			cbin = c_rot + descr_width / 2 - 0.5;

			if( rbin > -1.0  &&  rbin <descr_width  &&  cbin > -1.0  &&  cbin < descr_width )
				//旋转后邻域内采样点重新计算梯度大小和方向
				if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
				{
					grad_ori -= ori;
					while( grad_ori < 0.0 )
						grad_ori += PI2;
					while( grad_ori >= PI2 )
						grad_ori -= PI2;

					obin = grad_ori * bins_per_rad;
					w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
					interp_hist_entry( rbin, cbin, obin, grad_mag * w );
				}
		}
}

void CSift_Feathures::normalize_descr( struct feature* feat )
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat->d;

	for( i = 0; i < d; i++ )
	{
		cur = feat->descr[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt( len_sq );
	for( i = 0; i < d; i++ )
		feat->descr[i] *= len_inv;
}

void CSift_Feathures::hist_to_descr(  struct feature* feat )
{
	int int_val, i, r, c, o, k = 0;

	for( r = 0; r < descr_width; r++ )
		for( c = 0; c < descr_width; c++ )
			for( o = 0; o < descr_hist_bins; o++ )
				feat->descr[k++] = hist[r][c][o];

	feat->d = k;
	normalize_descr( feat );
	for( i = 0; i < k; i++ )
		if( feat->descr[i] > 0.2 )
			feat->descr[i] = 0.2;
	normalize_descr( feat );


	for( i = 0; i < k; i++ )
	{
		int_val = 512.0 * feat->descr[i];
		feat->descr[i] = MIN( 255, int_val );
	}
}

void CSift_Feathures::release_descr_hist( double ****phist)
{
	int i, j;

	for( i = 0; i < descr_width; i++)
	{
		for( j = 0; j < descr_width; j++ )
			free( (*phist)[i][j] );
		free( (*phist)[i] );
	}
	free( *phist );
	*phist = NULL;
}

void CSift_Feathures::release_pyr( vector<vector<Mat>> &pyr, int octvs, int n )
{
	int i, j;
	for( i = 0; i < octvs; i++ )
		for( j = 0; j < n; j++ )
			pyr[i][j].release();
}

void CSift_Feathures::calc_feature_oris()
{
	struct feature* feat;
	struct detection_data* ddata;
	double* hist;
	double omax;
	int i, j, n = featuresSeq->total;

	for( i = 0; i < n; i++ )
	{
		feat = (struct feature*)malloc( sizeof( struct feature ) );
		cvSeqPopFront( featuresSeq, feat );
		ddata = feat_detection_data( feat );
		hist = ori_hist( gauss_pyr[ddata->octv][ddata->intvl],
			ddata->r, ddata->c, 36,//每10°一个柱，共36柱
			cvRound( 4.5 * ddata->scl_octv ),//邻域窗口半径为3*1.5octv;
			1.5 * ddata->scl_octv );//梯度模值为1.5octv
		for( j = 0; j < 2; j++ )
			smooth_ori_hist( hist, 36 );
		omax = dominant_ori( hist, 36 );
		add_good_ori_features( hist, 36,
			omax * 0.8, feat );
		free( ddata );
		free( feat );
		free( hist );
	}
	// cout<<"   插值添加幅方向后特征点个数："<< features->total <<"  "<<(float)(num-aa+bb+cc+dd+ee)/(aa-bb-cc-dd-ee)<<endl;
}

void CSift_Feathures::compute_descriptors()
{
	struct feature* feat;
	struct detection_data* ddata;
	int i, k = featuresSeq->total;
	for( i = 0; i < k; i++ )
	{
		feat = CV_GET_SEQ_ELEM( struct feature, featuresSeq, i );
		ddata = feat_detection_data( feat );
		descr_hist( gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
			ddata->c, feat->ori, ddata->scl_octv );
		hist_to_descr( feat );
		release_descr_hist(&hist);
	}
}

int feature_cmp( void* feat1, void* feat2, void* param )
{
	struct feature* f1 = (struct feature*) feat1;
	struct feature* f2 = (struct feature*) feat2;

	if( f1->scl < f2->scl )
		return 1;
	if( f1->scl > f2->scl )
		return -1;
	return 0;
}

void CSift_Feathures::_sift_features(Mat img)
{

	int i;
	imgsrc = create_init_img(img);
	octvs = log( (float)MIN( imgsrc.cols, imgsrc.rows ) ) / log(2.0) - 2;
	build_gauss_pyr();
	build_dog_pyr();
	storage = cvCreateMemStorage( 0 );
	scale_space_extrema();
	calc_feature_scales();//计算特征尺度
	if( img_dbl )
		adjust_for_img_dbl();
	calc_feature_oris(  );//规划特征向量的方向
	compute_descriptors( );//计算描述子
	//按特诊点的尺度对特征描述向量进行排序
	cvSeqSort( featuresSeq, (CvCmpFunc)feature_cmp, NULL );
	n = featuresSeq->total;

	cout<<"图片1的特征点："<<n<<endl;
	//features = new feature[n];
	features =(struct feature *) calloc( n, sizeof(struct feature) );
	//制序列的全部或部分到一个连续内存数组中
	features =(feature*)cvCvtSeqToArray( featuresSeq, features, CV_WHOLE_SEQ );
	for( i = 0; i < n; i++ )
	{
		free( features[i].feature_data );
		features[i].feature_data = NULL;
	}

	export_features( "1281.txt", features, n );

}

void CSift_Feathures::export_features(string filename, struct feature* feat,int n)
{

	//string filename = "2.xml";
	int d = feat[0].d;    //128
	FileStorage fs(filename, FileStorage::WRITE);

	Mat R=Mat(n,d,CV_8UC1);
	for( int i = 0; i < n; i++ )
	{
		//fs<<"strings"<<"[";
		//fs<<"]";
		//fs<<"kkk"<<feat[i].y; //" b"<<feat[i].x<<" c "<<feat[i].scl<<" d "<<feat[i].ori<<endl ;
		
		for(int j = 0; j < d; j++ )
		{					
			R.at<uchar>(i,j) = feat[i].descr[j];
		}
	}

	fs << "R" << R;                                     
	fs.release();	
}
/*
int CSift_Feathures::export_features( char* filename, struct feature* feat, int n )
{
	if( n <= 0  ||  ! feat )
	{
		fprintf( stderr, "Warning: no features to export, %s line %d\n",
			__FILE__, __LINE__ );
		return 1;
	}
	FILE* file;
	int i, j, d;

	if( ! ( file = fopen( filename, "w" ) ) )
	{
		fprintf( stderr, "Warning: error opening %s, %s, line %d\n",
			filename, __FILE__, __LINE__ );
		return 1;
	}

	d = feat[0].d;
	fprintf( file, "%d %d\n", n, d );
	for( i = 0; i < n; i++ )
	{
		fprintf( file, "%f %f %f %f", feat[i].y, feat[i].x,
			feat[i].scl, feat[i].ori );
		for( j = 0; j < d; j++ )
		{			
			if( j % 20 == 0 )
				fprintf( file, "\n" );
			fprintf( file, " %d", (int)(feat[i].descr[j]) );
		}
		fprintf( file, "\n" );
	}

	if( fclose(file) )
	{
		fprintf( stderr, "Warning: file close error, %s, line %d\n",
			__FILE__, __LINE__ );
		return 1;
	}

	return 0;
}

*/