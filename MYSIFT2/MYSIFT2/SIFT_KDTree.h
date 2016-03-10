#pragma once


#include "SIFT_Features.h"
//#include <opencv2/core/core_c.h>


struct kd_node
{
	int ki;                      /**< partition key index */
	double kv;                   /**< partition key value */
	int leaf;                    /**< 1 if node is a leaf, 0 otherwise */
	struct feature* features;    /**< features at this node */
	int n;                       /**< number of features */
	struct kd_node* kd_left;    /**< left child */
	struct kd_node* kd_right;    /**< right child */
};


class CSift_KD_Tree
{
public:
	CSift_KD_Tree(void);
	~CSift_KD_Tree(void);
	struct kd_node* kd_node_init( struct feature*, int );
	void expand_kd_node_subtree( struct kd_node* );
	void assign_part_key( struct kd_node* );
	double median_select( double*, int );
	double rank_select( double*, int, int );
	void insertion_sort( double*, int );
	int partition_array( double*, int, double );
	void partition_features( struct kd_node* );




	int within_rect( CvPoint2D64f, CvRect );
	struct kd_node* kdtree_build( struct feature* features, int n );
public:
	struct kd_node *kd_root;
};




