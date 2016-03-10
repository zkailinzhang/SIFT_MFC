#include "StdAfx.h"
#include "SIFT_KDTree.h"
#include "SIFT_Matches.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

CSift_Matchs::CSift_Matchs(void)
{
}

CSift_Matchs::~CSift_Matchs(void)
{
}

struct min_pq* minpq_init()
{
	struct min_pq* min_pq;


	//min_pq = new struct min_pq;

	min_pq = (struct min_pq* )malloc( sizeof( struct min_pq ) );
	min_pq->pq_array = (struct pq_node*)calloc( MINPQ_INIT_NALLOCD, sizeof( struct pq_node ) );
	min_pq->nallocd = MINPQ_INIT_NALLOCD;
	min_pq->n = 0;

	return min_pq;
}


double CSift_Matchs::descr_dist_sq( struct feature* f1, struct feature* f2 )
{
	double diff, dsq = 0;
	double* descr1, * descr2;
	int i, d;

	d = f1->d;
	if( f2->d != d )
		return DBL_MAX;
	descr1 = f1->descr;
	descr2 = f2->descr;

	for( i = 0; i < d; i++ )
	{
		diff = descr1[i] - descr2[i];
		dsq += diff*diff;
	}
	return dsq;
}

Mat CSift_Matchs::stack_imgs(Mat &img1,Mat &img2){

	Mat stacked(img1.rows+ img2.rows+2,MAX(img1.cols,img2.cols),CV_8UC3,Scalar::all(0));

	Mat roi1 = stacked(Range(0,img1.rows),Range(0,img1.cols));
	img1.copyTo(roi1);

	Mat roi2 = stacked(Range(img1.rows+2,img1.rows+img2.rows+2),Range(0,img2.cols));	
	img2.copyTo(roi2);
    
//	imshow("a",stacked);
//	waitKey(0);

	return stacked;
}

void CSift_Matchs::match(struct feature *feat1,int n1,struct feature *feat2,int n2,Mat &img1,Mat &stacked){
	
	kd_tree.kd_root = kd_tree.kdtree_build(feat2,n2);
	struct feature *feat;
	struct feature **nbrs;//���������ھӵ�����
	int k=2;
	Point pt1, pt2;
	double d0, d1;
	int num=0;
	for (int i=0;i<n1;i++)
	{
		feat=feat1+i;

		k=kdtree_bbf_knn(kd_tree.kd_root,feat,2,&nbrs,200);

		if (k==2)
		{
			d0=descr_dist_sq(feat,nbrs[0]);
			d1=descr_dist_sq(feat,nbrs[1]);
			if (d0<d1* 0.2)
			{
				pt1=Point(cvRound(feat->x),cvRound(feat->y));
				pt2=Point(cvRound(nbrs[0]->x),cvRound(nbrs[1]->y));
				pt2.y+=(img1.rows+2);
				line(stacked,pt1,pt2,CV_RGB(255,0,255),1,8,0);
				feat1[i].fwd_match=nbrs[0];
				num++;
			}
		}
		free(nbrs);

	}


// 	imshow("aaa",stacked);
// 	waitKey(0);
	cout<<"ƥ����Ŀ��"<<num<<endl;
}



/*����һ�������㵽��������飬ʹ�����еĵ㰴��Ŀ���ľ�����������
������
feat��Ҫ����������㣬��feature_data��Ӧ��ָ��bbf_data�ṹ��ָ�룬���е�dֵʱfeat��Ŀ���ľ����ƽ��
nbrs�����������
n����������������е�Ԫ�ظ���
k�����������Ԫ�ظ��������ֵ
����ֵ����feat�ɹ����룬����1�����򷵻�0
*/
int CSift_Matchs::insert_into_nbr_array( struct feature* features, struct feature** nbrs,
	int n, int k )
{	
	//fdata��Ҫ����ĵ��bbf�ṹ��ndata������������еĵ��bbf�ṹ
	struct bbf_data* fdata, * ndata;
	//dn��������������������bbf�ṹ�еľ���ֵ��df��Ҫ������������bbf�ṹ�еľ���ֵ
	double dn, df;
	int i, ret = 0;

	if( n == 0 )
	{
		nbrs[0] = features;
		return 1;
	}

	
	fdata = (struct bbf_data*)features->feature_data;
	df = fdata->d;//Ҫ������������bbf�ṹ�еľ���ֵ	
	//����������еĵ��bbf�ṹ
	ndata = (struct bbf_data*)nbrs[n-1]->feature_data;
	//��������������һ���������bbf�ṹ�еľ���ֵ
	dn = ndata->d;
	if( df >= dn )
	{
		if( n == k )
		{
			features->feature_data = fdata->old_data;
			free( fdata );
			return 0;
		}
		nbrs[n] = features;
		return 1;
	}
	if( n < k )
	{
		nbrs[n] = nbrs[n-1];
		ret = 1;
	}
	else
	{
		nbrs[n-1]->feature_data = ndata->old_data;
		free( ndata );
	}
	i = n-2;
	while( i >= 0 )
	{
		ndata = (struct bbf_data*)nbrs[i]->feature_data;
		dn = ndata->d;
		if( dn <= df )
			break;
		nbrs[i+1] = nbrs[i];
		i--;
	}
	i++;
	nbrs[i] = features;

	return ret;
}



struct min_pq* CSift_Matchs::min_pq_init()
{
	struct min_pq *min_pq;

	min_pq = (struct min_pq*)malloc(sizeof(struct min_pq));

	min_pq->pq_array = (struct pq_node*)calloc(MINPQ_INIT_NALLOCD,sizeof(struct pq_node*));
	min_pq->nallocd = MINPQ_INIT_NALLOCD;
	min_pq->n = 0;
	return min_pq;
}


/**
*����С�����ȼ������в���һ����Ա
*@param min_pq һ����С�������ȼ�����
*@param data ��Ҫ��������ݳ�Ա
*@param key ���ݳ�Ա�Ĺؼ�ֵ
*@return �ɹ�����0���򷵻�1
*/
int CSift_Matchs::minpq_insert( struct min_pq* min_pq, void* data, int key)
{
	int n = min_pq->n; //0

	if (min_pq->nallocd == n)   //512  0  �������ݳ����������ڴ�
	{
		min_pq->nallocd  = array_double( &min_pq->pq_array, min_pq->nallocd,sizeof( struct pq_node* ));
		if (!min_pq->nallocd)
		{
			fprintf( stderr, "Warning: unable to allocate memory, %s, line %d\n");
			return 1;
		}
	}

	min_pq->pq_array[n].data = data;

	min_pq->pq_array[n].key = INT_MAX;

	decrease_pq_node_key(min_pq->pq_array, min_pq->n, key);      //???

	min_pq->n++;

	return 0;
}


/*
  Decrease a minimizing pq element's key, rearranging the pq if necessary

  @param pq_array minimizing priority queue array
  @param i index of the element whose key is to be decreased
  @param key new value of element <EM>i</EM>'s key; if greater than current
    key, no action is taken
*/
void CSift_Matchs::decrease_pq_node_key( struct pq_node* pq_array, int i , int key )
{
	struct pq_node tmp;

	if (key > pq_array[i].key)
	{
		return;
	}
	pq_array[i].key = key;

	while(i>0 && pq_array[i].key < pq_array[int(( i - 1 ) / 2)].key)
	{
		tmp = pq_array[int(( i - 1 ) / 2)];
		pq_array[int(( i - 1 ) / 2)] = pq_array[i];
		pq_array[i] = tmp;
		i = int(( i - 1 ) / 2);
	}
}


void* CSift_Matchs::minpq_extract_min(struct min_pq *min_pq)
{
	void* data;

	if( min_pq->n < 1 )
	{
		fprintf( stderr, "Warning: PQ empty, %s line %d\n", __FILE__, __LINE__ );
		return NULL;
	}
	data = min_pq->pq_array[0].data;
	min_pq->n--;
	min_pq->pq_array[0] = min_pq->pq_array[min_pq->n];
	restore_minpq_order( min_pq->pq_array, 0, min_pq->n );

	return data;
}

/*
  Recursively restores correct priority queue order to a minimizing pq array

  @param pq_array a minimizing priority queue array
  @param i index at which to start reordering
  @param n number of elements in \a pq_array
*/
void CSift_Matchs::restore_minpq_order( struct pq_node* pq_array, int i, int n)
{
	struct pq_node tmp;
	int l,r,min = i;
	l = (2*i)+1;
	r = (2*i)+2;
	if (l<n)
	{
		if(pq_array[l].key<pq_array[i].key)
			min = l;
	}
	if (r<n)
	{
		if (pq_array[r].key < pq_array[min].key)
		{
			min = r;
		}
	}
	if (min !=i)
	{
		tmp = pq_array[min];
		pq_array[min] = pq_array[i];
		pq_array[i] = tmp;
		restore_minpq_order(pq_array,min,n);
	}
}
void CSift_Matchs::minpq_release(struct min_pq* min_pq)
{
	if( ! min_pq )
	{
		fprintf( stderr, "Warning: NULL pointer error, %s line %d\n", __FILE__,
			__LINE__ );
		return;
	}
	if( min_pq  &&  min_pq->pq_array )
	{
		free( min_pq->pq_array );
		free( min_pq );
		min_pq = NULL;
	}
}


/*�����ӽ������k-d��ֱ��Ҷ�ڵ㣬���������н�δ�����Ľڵ�������ȼ��������

���ȼ����к�����·����ͬʱ���ɵģ���Ҳ��BBF�㷨�ľ������ڣ��ڶ���������ʱ
������·����һ��ķ�֧���뵽���ȼ������У�������ʱ���ҡ������ȼ����е���
����Ǹ���Ŀ��������ָƽ��ľ���ABS(kv - feat->descr[ki])
������
kd_node��Ҫ����������������
feat��Ŀ��������
min_pq�����ȼ�����
����ֵ��Ҷ�ӽڵ��ָ��
*/
struct kd_node* CSift_Matchs::explore_to_leaf( struct kd_node* kd_node, struct feature* feat,struct min_pq* min_pq )
{
	struct kd_node* unexpl, * expl = kd_node;
	//unexpl�д�������ȼ����еĺ�ѡ�ڵ�(��δ�����Ľڵ�)��explΪ��ǰ�����ڵ�
	double kv;
	int ki;
	//һֱ������Ҷ�ӽڵ㣬���������н�δ�����Ľڵ�������ȼ��������
	while( expl  &&  ! expl->leaf )
	{
		ki = expl->ki;
		kv = expl->kv;

		if( ki >= feat->d )
		{
			fprintf( stderr, "Warning: comparing imcompatible descriptors, %s" \
				" line %d\n", __FILE__, __LINE__ );
			return NULL;
		}
		//Ŀ���kiά����<=kv�������������������������������ȶ���
		if( feat->descr[ki] <= kv )
		{
			unexpl = expl->kd_right;
			expl = expl->kd_left;
		}
		else
		{
			unexpl = expl->kd_left;
			expl = expl->kd_right;
		}

		 //����ѡ�ڵ�unexpl����Ŀ��������kiά���丸�ڵ�ľ�����뵽���ȶ����У�����ԽС�����ȼ�Խ��
		if( minpq_insert( min_pq, unexpl, ABS( kv - feat->descr[ki] ) ) )
		{
			fprintf( stderr, "Warning: unable to insert into PQ, %s, line %d\n",
				__FILE__, __LINE__ );
			return NULL;
		}
	}

	return expl;  //����Ҷ�ӽڵ��ָ��
}


void CSift_Matchs::kdtree_release(struct kd_node* kd_root)
{
	if (!kd_root)
	{
		return;
	}
	kdtree_release(kd_root->kd_left);
	kdtree_release(kd_root->kd_right);
	free(kd_root);
}



int  CSift_Matchs::kdtree_bbf_knn( struct kd_node* kd_root, struct feature* features, 
	int k, struct feature*** nbrs, int max_nn_chks )
{
	struct kd_node *expl; //��ǰ���ҵ�
	struct min_pq *min_pq; //���ȼ�����
	struct feature *tree_feat, **_nbrs;//tree_feat�ǵ���SIFT������_nbrs�д���Ų��ҳ����Ľ��������ڵ�
	struct bbf_data *bbf_data;//bbf_data��һ�����������ʱ�������ݺ����������Ļ���ṹ
	int i,t=0, n=0;           //t����������������n�ǵ�ǰ����������е�Ԫ�ظ���
	if (!nbrs || !features || !kd_root)
	{
		fprintf( stderr, "Warning: NULL pointer error, %s, line %d\n");
		return -1;
	}
	_nbrs = (struct feature**)calloc(k,sizeof(struct feature*));
	min_pq = minpq_init();
	minpq_insert( min_pq, kd_root, 0);//�����ڵ��Ȳ��뵽min_pq���ȼ�������
	while(min_pq->n >0 && t<max_nn_chks)
	{
		expl = (struct kd_node*)minpq_extract_min( min_pq );//��min_pq����ȡ(���Ƴ�)���ȼ���ߵĽڵ㣬��ֵ����ǰ�ڵ�expl
		if( ! expl )
		{
			fprintf( stderr, "Warning: PQ unexpectedly empty, %s line %d\n",
				__FILE__, __LINE__ );
			goto fail;
		}	
		expl = explore_to_leaf( expl, features, min_pq );//�Ӹõ㿪ʼ��explore��leaf��·���ġ�������ĵ㡱��������С����min_pq�С�
		if( ! expl )
		{
			fprintf( stderr, "Warning: PQ unexpectedly empty, %s line %d\n",
				__FILE__, __LINE__ );
			goto fail;
		}

		for( i = 0; i < expl->n; i++ )
		{
			tree_feat = &expl->features[i];
			bbf_data = (struct bbf_data *)malloc( sizeof( struct bbf_data ) );
			if( ! bbf_data )
			{
				fprintf( stderr, "Warning: unable to allocate memory,"
					" %s line %d\n", __FILE__, __LINE__ );
				goto fail;
			}
			bbf_data->old_data = tree_feat->feature_data;//�����i���������feature_data����ǰ��ֵ
			bbf_data->d = descr_dist_sq(features, tree_feat);//��ǰ�������Ŀ���֮���ŷ�Ͼ���
			tree_feat->feature_data = bbf_data;
			//�жϲ�������������������㵽���������_nbrs��,����ɹ�����1
			//�������������Ԫ�ظ����Ѵﵽkʱ����������Ԫ�ظ����������ӣ��������Ԫ�ص�ֵ
			n += insert_into_nbr_array( tree_feat, _nbrs, n, k );
		}
		t++;
	}
	minpq_release( min_pq );

	//��������������е������㣬�ָ���feature_data���ֵ
	for( i = 0; i < n; i++ )
	{	
		bbf_data = (struct bbf_data*)(_nbrs[i]->feature_data);
		_nbrs[i]->feature_data = bbf_data->old_data;
		free( bbf_data );
	}
	*nbrs = _nbrs;
	return n;

fail:
	minpq_release( min_pq );
	for( i = 0; i < n; i++ )
	{
		bbf_data = (struct bbf_data*)(_nbrs[i]->feature_data);
		_nbrs[i]->feature_data = bbf_data->old_data;
		free( bbf_data );
	}
	free( _nbrs );
	*nbrs = NULL;
	return -1;
}

int CSift_Matchs::array_double( struct pq_node** array, int n, int size )
{
	struct pq_node* tmp;

	tmp = (struct pq_node*)realloc( *array, 2 * n * size );
	if( ! tmp )
	{
		fprintf( stderr, "Warning: unable to allocate memory in array_double(),"
			" %s line %d\n", __FILE__, __LINE__ );
		if( *array )
			free( *array );
		*array = NULL;
		return 0;
	}
	*array = tmp;
	return n*2;
}