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
	//�Ҷ�ת��
	Mat convert_to_gray32(Mat &img );
	//������������0��ԭʼͼ��(�Ŵ�2��)
	Mat create_init_img(Mat img);
	//���²���
	Mat downsample(Mat img );
	//������˹������
	void build_gauss_pyr(void);
	//������˹��ֽ�����
	void build_dog_pyr(void);
	//����ĳ��ĳ�е�����ֵ
	float pixval32f(Mat img, int r, int c );
	//�ж��Ƿ��Ǽ�ֵ��(��26���ٽ���������С��)
	int is_extremum(int octv, int intvl,int r, int c );
	//����ά(x,y,sigma)ƫ����
	CvMat* deriv_3D(  int octv, int intvl, int r, int c );
	//����ά���ε���(��ɭ����)
	CvMat* hessian_3D( int octv, int intvl, int r, int c );
	//��ȡ�����ؼ�ֵ���λ��
	struct feature* interp_extremum(  int octv, int intvl,int r, int c);
	//��ȡ�����ؼ�ֵ���λ�����õ��ĺ���
	void interp_step(  int octv, int intvl, int r, int c,double* xi, double* xr, double* xc );
	//����������صĶԱȶ�
	double interp_contr(  int octv, int intvl, int r,int c, double xi, double xr, double xc );
	//������ֵ��
	struct feature* new_feature( void );
	//������Ե��Ӧ
	int is_too_edge_like( Mat dog_img, int r, int c);

	//1����ֵ�����Լ��ؼ��㶨λ������Ե��Ӧ
	void scale_space_extrema();

	//2�����������߶�
	void calc_feature_scales( );
	//����ͼ����������,�߶�,��������СΪԭ����һ��
	void adjust_for_img_dbl();


	//3:�����������ص��ݶȴ�С�ͷ���
	int calc_grad_mag_ori( Mat img, int r, int c, double* mag, double* ori );

	//���������ؼ���Ҷ�ֱ��ͼ
	double* ori_hist( Mat img, int r, int c, int n, int rad, double sigma);
	//��ֱ��ͼ���и�˹ģ��
	void smooth_ori_hist( double* hist, int n );
	//3����ֱ��ͼ���ҵ��������ݶ�
	double dominant_ori( double* hist, int n );


	//������������
	struct feature* clone_feature( struct feature* feat );
	//������ĳһ���ݶȴ�С����ֵ�������������뵽ֱ��ͼ
	void add_good_ori_features( double* hist, int n,double mag_thr, struct feature* feat );
	//����һ��entry���뵽����ֱ��ͼ�дӶ��γ�����������
	void interp_hist_entry(  double rbin, double cbin,
		double obin, double mag );
	//�����άֱ��ͼ
	void descr_hist(Mat img, int r, int c, double ori,
		double scl);
	//��һ��������
	void normalize_descr( struct feature* feat );
	//����άֱ��ͼת��Ϊ����������
	void hist_to_descr( struct feature* feat );
	void release_descr_hist(double ****phist);
	void release_pyr(  vector<vector<Mat>>& pyr, int octvs, int n );

	//��ÿһ��ͼ��������������淶���ķ���
	void calc_feature_oris( );
	//4����������������
	void compute_descriptors();

	//��sift����������ȡ����
	void _sift_features(Mat img);

	void export_features( string filename, struct feature* feat, int n );
public:
	vector<vector<Mat>> gauss_pyr;
	vector<vector<Mat>> dog_pyr;
	struct feature *features;
	int intvls;//��˹������S(һ��ΪS+3)
	double contr_thr;
	int curv_thr;
	double sigma;
	int img_dbl;
	Mat imgsrc;
	int descr_width;
	int descr_hist_bins;
	int octvs;
	CvMemStorage* storage;//�ڴ�洢��
	CvSeq* featuresSeq;
	int n;          //���������
	double ***hist;//ֱ��ͼָ��

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
    int type;                      /**< feature type, OXFD or LOWE ������ ����*/  
    int category;                  /**< all-purpose feature category */  
    struct feature* fwd_match;     /**< matching feature from forward image */  
    struct feature* bck_match;     /**< matching feature from backmward image */  
    struct feature* mdl_match;     /**< matching feature from model */  
    CvPoint2D64f img_pt;           /**< location in image */  
    CvPoint2D64f mdl_pt;           /**< location in model */  
    void* feature_data;            /**< user-definable data */  
    char dense;                     /*�����������������̶ܳ�*/  
	//int n;                         //���������Ӹ���
};  



//��ֵ�������õ��Ľṹ
//��SIFT������ȡ�����У����������ݻᱻ��ֵ��feature�ṹ��feature_data��Ա

struct detection_data
{
	int r;      //���������ڵ���
	int c;      //���������ڵ���
	int octv;   //��˹��ֽ������У����������ڵ���
	int intvl;  //��˹��ֽ������У����������ڵ����еĲ�
	double subintvl;  //�������ڲ㷽��(�ҷ���,intvl����)�ϵ�������ƫ����
	double scl_octv;  //���������ڵ���ĳ߶�
};

