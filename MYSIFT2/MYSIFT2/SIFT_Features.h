#pragma once

#include <cstring>
#include <opencv2\opencv.hpp>

using namespace cv;

#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )
#ifndef ABS
#define ABS(x) ( ( x < 0 )? -x : x )
#endif
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )
//??
#define FEATURE_MAX_D 132  

class CSift_Feathures
{
public:
	CSift_Feathures(void);
	~CSift_Feathures(void);
	//灰度转换
	Mat convert_to_gray32(Mat &img );
	//创建金字塔第0层原始图像(放大2倍)
	Mat create_init_img(Mat img);
	//向下采样
	Mat downsample(Mat img );
	//创建高斯金字塔
	void build_gauss_pyr(void);
	//创建高斯差分金字塔
	void build_dog_pyr(void);
	//返回某行某列的像素值
	float pixval32f(Mat img, int r, int c );
	//判断是否是极值点(与26个临界点找最大最小点)
	int is_extremum(int octv, int intvl,int r, int c );
	//求三维(x,y,sigma)偏导数
	CvMat* deriv_3D(  int octv, int intvl, int r, int c );
	//求三维二次倒数(海森矩阵)
	CvMat* hessian_3D( int octv, int intvl, int r, int c );
	//获取亚像素极值点的位置
	struct feature* interp_extremum(  int octv, int intvl,int r, int c);
	//获取亚像素极值点的位置所用到的函数
	void interp_step(  int octv, int intvl, int r, int c,double* xi, double* xr, double* xc );
	//计算插入像素的对比度
	double interp_contr(  int octv, int intvl, int r,int c, double xi, double xr, double xc );
	//构建极值点
	struct feature* new_feature( void );
	//消除边缘响应
	int is_too_edge_like( Mat dog_img, int r, int c);

	//1：极值点检测以及关键点定位消除边缘响应
	void scale_space_extrema();

	//2：计算特征尺度
	void calc_feature_scales( );
	//调整图像特征坐标,尺度,点的坐标大小为原来的一半
	void adjust_for_img_dbl();


	//3:计算所给像素的梯度大小和方向
	int calc_grad_mag_ori( Mat img, int r, int c, double* mag, double* ori );

	//给所给像素计算灰度直方图
	double* ori_hist( Mat img, int r, int c, int n, int rad, double sigma);
	//对直方图进行高斯模糊
	void smooth_ori_hist( double* hist, int n );
	//3：在直方图中找到主方向梯度
	double dominant_ori( double* hist, int n );


	//拷贝特征向量
	struct feature* clone_feature( struct feature* feat );
	//将大于某一个梯度大小的阈值的特征向量加入到直方图
	void add_good_ori_features( double* hist, int n,double mag_thr, struct feature* feat );
	//插入一个entry进入到方向直方图中从而形成特征描述子
	void interp_hist_entry(  double rbin, double cbin,
		double obin, double mag );
	//计算二维直方图
	void descr_hist(Mat img, int r, int c, double ori,
		double scl);
	//归一化描述子
	void normalize_descr( struct feature* feat );
	//将二维直方图转换为特征描述子
	void hist_to_descr( struct feature* feat );
	void release_descr_hist(double ****phist);
	void release_pyr(  vector<vector<Mat>>& pyr, int octvs, int n );

	//给每一个图像特征向量计算规范化的方向
	void calc_feature_oris( );
	//4：计算特征描述子
	void compute_descriptors();

	//主sift特征向量提取程序
	void _sift_features(Mat img);

	void export_features( string filename, struct feature* feat, int n );
public:
	vector<vector<Mat>> gauss_pyr;
	vector<vector<Mat>> dog_pyr;
	struct feature *features;
	int intvls;//高斯金字塔S(一般为S+3)
	double contr_thr;
	int curv_thr;
	double sigma;
	int img_dbl;
	Mat imgsrc;
	int descr_width;
	int descr_hist_bins;
	int octvs;
	CvMemStorage* storage;//内存存储块
	CvSeq* featuresSeq;
	int n;          //特征点个数
	double ***hist;//直方图指针

};

/** 
Structure to represent an affine invariant image feature.  The fields 
x, y, a, b, c represent the affine region around the feature: 
 
a(x-u)(x-u) + 2b(x-u)(y-v) + c(y-v)(y-v) = 1 
*/  
struct feature  
{  
    double x;                      /**< x coord */  
    double y;                      /**< y coord */  
    double a;                      /**< Oxford-type affine region parameter */  
    double b;                      /**< Oxford-type affine region parameter */  
    double c;                      /**< Oxford-type affine region parameter */  
    double scl;                    /**< scale of a Lowe-style feature */  
    double ori;                    /**< orientation of a Lowe-style feature */  
    int d;                         /**< descriptor length */  
    double descr[FEATURE_MAX_D];   /**< descriptor */  
    int type;                      /**< feature type, OXFD or LOWE 画特征 两种*/  
    int category;                  /**< all-purpose feature category */  
    struct feature* fwd_match;     /**< matching feature from forward image */  
    struct feature* bck_match;     /**< matching feature from backmward image */  
    struct feature* mdl_match;     /**< matching feature from model */  
    CvPoint2D64f img_pt;           /**< location in image */  
    CvPoint2D64f mdl_pt;           /**< location in model */  
    void* feature_data;            /**< user-definable data */  
    char dense;                     /*表征特征点所处稠密程度*/  
	//int n;                         //特征描述子个数
};  



//极值点检测中用到的结构
//在SIFT特征提取过程中，此类型数据会被赋值给feature结构的feature_data成员

struct detection_data
{
	int r;      //特征点所在的行
	int c;      //特征点所在的列
	int octv;   //高斯差分金字塔中，特征点所在的组
	int intvl;  //高斯差分金字塔中，特征点所在的组中的层
	double subintvl;  //特征点在层方向(σ方向,intvl方向)上的亚像素偏移量
	double scl_octv;  //特征点所在的组的尺度
};

