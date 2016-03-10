#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;


#define MINPQ_INIT_NALLOCD 512


struct min_pq
{
	struct pq_node* pq_array;    
	int nallocd;                 
	int n;                       
};

struct pq_node
{
	void* data;
	int key;
};


struct bbf_data
{
	double d;
	void* old_data;
};

/**
   Doubles the size of an array with error checking

   @param array pointer to an array whose size is to be doubled
   @param n number of elements allocated for \a array
   @param size size in bytes of elements in \a array

   @return Returns the new number of elements allocated for \a array.  If no
     memory is available, returns 0 and frees \a array.
*/



//extern int array_double( struct pq_node**, int n, int size );



class CSift_Matchs
{
public:
	CSift_Matchs(void);
	~CSift_Matchs(void);

	void match(struct feature *feat1,int n1,struct feature *feat2,int n2,Mat &src,Mat &img2);
	double descr_dist_sq( struct feature* f1, struct feature* f2 );
	Mat stack_imgs(Mat &img1,Mat &img2);


	int insert_into_nbr_array( struct feature*, struct feature**, int, int );
	int kdtree_bbf_knn( struct kd_node* kd_root, struct feature* feat,
	int k, struct feature*** nbrs, int max_nn_chks );
	
	void kdtree_release( struct kd_node* kd_root );

	void minpq_release(struct min_pq* min_pq);
	struct min_pq* min_pq_init();  //Œ¥”√

	struct kd_node* explore_to_leaf( struct kd_node*, struct feature*,struct min_pq* );
	int minpq_insert( struct min_pq* , void* , int );
	
	void decrease_pq_node_key( struct pq_node*, int, int );
	void* minpq_extract_min(struct min_pq *);
	void restore_minpq_order( struct pq_node*, int, int );
	int array_double( struct pq_node** , int , int );

	//int kdtree_bbf_spatial_knn( struct kd_node* kd_root,struct feature* feat, int k,struct feature*** nbrs, int max_nn_chks,CvRect rect, int model );



private:
	CSift_KD_Tree kd_tree;
	//CSift_Bbf_Knn bbf_knn;
};





//int match_points(struct feature* features,struct kd_node* kd_root,IplImage *img1,IplImage *img2,char*);
//	int array_double( struct pq_node** , int , int );


