#include "StdAfx.h"
#include "SIFT_KDTree.h"
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;




CSift_KD_Tree::CSift_KD_Tree(void)
{
}


CSift_KD_Tree::~CSift_KD_Tree(void)
{
}



struct kd_node* CSift_KD_Tree::kdtree_build(struct feature *features,int n)
{
	struct kd_node *kd_root;

	if (!features || n<=0)
	{
		fprintf( stderr, "Warning: create_kdtree(): no features, %s, line %d\n");
		return NULL;
	}
	//建立根节点
	kd_root = kd_node_init(features,n);
	//扩展根节点的左右子树
	expand_kd_node_subtree(kd_root);

	
	return kd_root;     //有了根节点 就有整棵树？   返回根节点

}

struct kd_node * CSift_KD_Tree::kd_node_init(struct feature* features,int n)
{
	struct kd_node *kd_node;

	kd_node = (struct kd_node *)malloc(sizeof(struct kd_node));
	memset(kd_node,0,sizeof(struct kd_node));

	kd_node->ki = 0;
	kd_node->features = features;
	kd_node->n = n;
	return kd_node;
}
void CSift_KD_Tree::expand_kd_node_subtree(struct kd_node *kd_node)
{
	if (kd_node->n ==1 || kd_node->n == 0)
	{
		kd_node->leaf = 1;
		return;
	}
	///*确定输入节点的枢轴索引和枢轴值
	assign_part_key(kd_node);
	//在指定k-d树节点上划分特征点集(即根据指定节点的ki和kv值来划分特征点集)
	partition_features(kd_node);
	if (kd_node->kd_left)
	{
		expand_kd_node_subtree(kd_node->kd_left);
	}
	if (kd_node->kd_right)
	{
		expand_kd_node_subtree(kd_node->kd_right);
	}
	
}
void CSift_KD_Tree::assign_part_key(struct kd_node *kd_node)
{
	struct feature *features;
	//枢轴的值kv，均值mean，方差var，方差最大值var_max
	double kv,x,mean,var,var_max=0;
	double *tmp;   
	int d,n,i,j,ki = 0;

	features = kd_node->features;	
	n = kd_node->n;     
	d = features[0].d;  //128

	//枢轴的索引值就是方差最大的那一维的维数,即n个128维的特征向量中，若第k维的方差最大，则k就是枢轴(分割位置)
	for (j=0;j<d;j++)
	{
		mean = var = 0;
		for (i=0;i<n;i++)
		{
			//cout<<"  "<<features[i].descr[j]<<endl;
			mean +=features[i].descr[j];
		}
		mean/=n;       // j维度的均值

		for (i=0;i<n;i++)
		{
			x = features[i].descr[j]-mean;
			var += x*x;
		}
		var/=n;    // j维度的方差

		if (var>var_max)
		{
			ki = j;
			var_max = var;
		}
	}
	//枢轴的值就是所有特征向量的ki维的中值(按ki维排序后中间的那个值)
	tmp = (double *)calloc(n,sizeof(double));


	for (i=0;i<n;i++)
	{
		tmp[i] = features[i].descr[ki];
	}


	//找到输入数组的中值
	kv = median_select(tmp,n);
	free(tmp);

	kd_node->ki = ki;
	kd_node->kv = kv;
}

double CSift_KD_Tree::median_select(double *array, int n)
{
	//找array数组中的第(n-1)/2小的数，即中值
	return rank_select(array, n, (n-1)/2);
}
double CSift_KD_Tree::rank_select(double *array,int n,int r)
{
	//将数组分成5个一组,在每组中找到中值，然后将中值合并继续找中值
	double  *tmp, med;
	/*
	int gr_5,gr_tot,rem_elts,i,j;
	gr_5 = n/5;
	gr_tot = cvCeil(n / 5.0);
	rem_elts = n % 5; //分组完成后，剩余元素*/
	tmp = array;
	insertion_sort(tmp,n);

	med = tmp[r];
	return med;
	/*
	for (i=0;i<gr_5;i++)
	{
		//插值排序
		insertion_sort(tmp,5);
		tmp +=5;
	}
	insertion_sort(tmp,rem_elts);
	tmp = (double *)calloc(gr_tot,sizeof(double));
	//将每个5元组中的中值(即下标为2,2+5,...的元素)复制到temp数组
	for (i=0,j=2;i<gr_5;i++,j+=5)
	{
		tmp[i] = array[j];
	}
	if (rem_elts)
	{
		tmp[i++] = array[n-1-rem_elts/2];
	}
	med = rank_select(tmp, i, (i-1)/2);
	free(tmp);
	
	//利用中值的中值划分数组，看划分结果是否是第r小的数，若不是则递归调用rank_select重新选择
	j = partition_array(array,n,med);
	if (r==j)
	{
		return med;
	} 
	else if (r<j)
	{
		return rank_select(array,j,r);
	} 
	else
	{
		array +=j+1;
		return rank_select(array,(n-j-1),(r-j-1));
	}
	*/
}
//插入排序  前面拍好的是正确的顺序 每一次拍完前面都是顺序的
void CSift_KD_Tree::insertion_sort(double *array ,int n)
{
	double k;
	int i,j;
	for (i=1;i<n;i++)
	{
		k = array[i];
		j = i-1;
		while(j>=0 && array[j]>k)
		{
			array[j+1] = array[i];
			j-=i;
		}
		array[j+1] = k;
	}
}

/*根据给定的枢轴值分割数组，使数组前部分小于pivot，后部分大于pivot
参数：
array：输入数组
n：数组的元素个数
pivot：枢轴值
返回值：分割后枢轴的下标
*/
int	CSift_KD_Tree::partition_array(double *array, int n,double pivot)	
{
	double tmp;
	int p,i,j;
	i = -1;
	for (j=0;j<n;j++)
	{
		if (array[j]<pivot)
		{
			tmp = array[++i];
			array[i] = array[j];
			array[j] = tmp;
			if (array[i] == pivot)
			{
				p=i;
			}
		}
	}
	array[p] = array[i];
	array[i] = pivot;

	return i;
}


/*在指定的k-d树节点上划分特征点集
使得特征点集的前半部分是第ki维小于枢轴的点，后半部分是第ki维大于枢轴的点
*/
void CSift_KD_Tree::partition_features(struct kd_node* kd_node)
{
	struct feature* features, tmp;                       //???????????????
	double kv;
	int n,ki,p,i,j=-1;

	features = kd_node->features;
	n = kd_node->n;
	ki = kd_node->ki;
	kv = kd_node->kv;


	for (i=0;i<n;i++)
	{
	 //若第i个特征点的特征向量的第ki维的值小于kv
	if( features[i].descr[ki] <= kv )
	{
		tmp = features[++j];
		features[j] = features[i];
		features[i] = tmp;
		if( features[j].descr[ki] == kv )
			p = j;//p保存枢轴所在的位置
	}
}
//将枢轴features[p]和最后一个小于枢轴的点features[j]对换
tmp = features[p];
features[p] = features[j];
features[j] = tmp;
//此后，枢轴的位置下标为j

//若所有特征点落在同一侧，则此节点成为叶节点
	if (j==n-1)
	{
		kd_node->leaf =1;
		return;
	}
	kd_node->kd_left = kd_node_init(features,j+1);
	kd_node->kd_right = kd_node_init(features+(j+1),(n-j-1));

}







