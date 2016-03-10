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
	//�������ڵ�
	kd_root = kd_node_init(features,n);
	//��չ���ڵ����������
	expand_kd_node_subtree(kd_root);

	
	return kd_root;     //���˸��ڵ� ������������   ���ظ��ڵ�

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
	///*ȷ������ڵ����������������ֵ
	assign_part_key(kd_node);
	//��ָ��k-d���ڵ��ϻ��������㼯(������ָ���ڵ��ki��kvֵ�����������㼯)
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
	//�����ֵkv����ֵmean������var���������ֵvar_max
	double kv,x,mean,var,var_max=0;
	double *tmp;   
	int d,n,i,j,ki = 0;

	features = kd_node->features;	
	n = kd_node->n;     
	d = features[0].d;  //128

	//���������ֵ���Ƿ���������һά��ά��,��n��128ά�����������У�����kά�ķ��������k��������(�ָ�λ��)
	for (j=0;j<d;j++)
	{
		mean = var = 0;
		for (i=0;i<n;i++)
		{
			//cout<<"  "<<features[i].descr[j]<<endl;
			mean +=features[i].descr[j];
		}
		mean/=n;       // jά�ȵľ�ֵ

		for (i=0;i<n;i++)
		{
			x = features[i].descr[j]-mean;
			var += x*x;
		}
		var/=n;    // jά�ȵķ���

		if (var>var_max)
		{
			ki = j;
			var_max = var;
		}
	}
	//�����ֵ������������������kiά����ֵ(��kiά������м���Ǹ�ֵ)
	tmp = (double *)calloc(n,sizeof(double));


	for (i=0;i<n;i++)
	{
		tmp[i] = features[i].descr[ki];
	}


	//�ҵ������������ֵ
	kv = median_select(tmp,n);
	free(tmp);

	kd_node->ki = ki;
	kd_node->kv = kv;
}

double CSift_KD_Tree::median_select(double *array, int n)
{
	//��array�����еĵ�(n-1)/2С����������ֵ
	return rank_select(array, n, (n-1)/2);
}
double CSift_KD_Tree::rank_select(double *array,int n,int r)
{
	//������ֳ�5��һ��,��ÿ�����ҵ���ֵ��Ȼ����ֵ�ϲ���������ֵ
	double  *tmp, med;
	/*
	int gr_5,gr_tot,rem_elts,i,j;
	gr_5 = n/5;
	gr_tot = cvCeil(n / 5.0);
	rem_elts = n % 5; //������ɺ�ʣ��Ԫ��*/
	tmp = array;
	insertion_sort(tmp,n);

	med = tmp[r];
	return med;
	/*
	for (i=0;i<gr_5;i++)
	{
		//��ֵ����
		insertion_sort(tmp,5);
		tmp +=5;
	}
	insertion_sort(tmp,rem_elts);
	tmp = (double *)calloc(gr_tot,sizeof(double));
	//��ÿ��5Ԫ���е���ֵ(���±�Ϊ2,2+5,...��Ԫ��)���Ƶ�temp����
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
	
	//������ֵ����ֵ�������飬�����ֽ���Ƿ��ǵ�rС��������������ݹ����rank_select����ѡ��
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
//��������  ǰ���ĺõ�����ȷ��˳�� ÿһ������ǰ�涼��˳���
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

/*���ݸ���������ֵ�ָ����飬ʹ����ǰ����С��pivot���󲿷ִ���pivot
������
array����������
n�������Ԫ�ظ���
pivot������ֵ
����ֵ���ָ��������±�
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


/*��ָ����k-d���ڵ��ϻ��������㼯
ʹ�������㼯��ǰ�벿���ǵ�kiάС������ĵ㣬��벿���ǵ�kiά��������ĵ�
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
	 //����i������������������ĵ�kiά��ֵС��kv
	if( features[i].descr[ki] <= kv )
	{
		tmp = features[++j];
		features[j] = features[i];
		features[i] = tmp;
		if( features[j].descr[ki] == kv )
			p = j;//p�����������ڵ�λ��
	}
}
//������features[p]�����һ��С������ĵ�features[j]�Ի�
tmp = features[p];
features[p] = features[j];
features[j] = tmp;
//�˺������λ���±�Ϊj

//����������������ͬһ�࣬��˽ڵ��ΪҶ�ڵ�
	if (j==n-1)
	{
		kd_node->leaf =1;
		return;
	}
	kd_node->kd_left = kd_node_init(features,j+1);
	kd_node->kd_right = kd_node_init(features+(j+1),(n-j-1));

}







