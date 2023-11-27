#ifndef NEIGHBOR_GRID_CUH
#define NEIGHBOR_GRID_CUH
#include "common.cuh"

//�Ե���ԭ�ӽ��пռ����񻮷֣��ڲ�ͬλ�ÿ�ͨ���������ӻ�õ��׽��ڱ�
struct NEIGHBOR_GRID
{
	NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket = NULL;
	INT_VECTOR grid_dimension;
	VECTOR box_length;
	VECTOR grid_length;
	VECTOR grid_length_inverse;
	float skin = 0.f;//��cutoff�Ļ���������һ������
	float skin2 = 0.f;
	float cutoff = 0.f;
	float cutoff2 = 0.f;
	int grid_numbers = 0;//��ά������ܴ�СNx*Ny*Nz
	int layer_numbers = 0;//һ�����ĿNx*Ny

	//ֻ�����ʼ��һ�Σ����ܷ�����ʼ��������Ҫ��
	//Ҫ��֤box_length�㹻������ȫ��ס��������
	//ע�����ֵ���������֣�����ĺ����ǽ����������������ڱ�����
	void Initial(VECTOR box_length, float cutoff, float skin);

	//��ÿ������ԭ�ӷŵ���ռ�������������
	//֮�󣬴������п�ֱ����ȡ��������ԭ�ӵı��
	void Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd);
	
	//��������ڵ�ԭ�Ӽ�¼
	void Clear_Grid_Bucket_Total();
	struct GPU
	{
		NEIGHBOR_GRID* neighbor_grid = NULL;

		NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket = NULL;

		//��CPU��ʵ�֣���������������GPU����Ϊ��������Թ̶�����ֻ����һ��
		void Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd);
	}gpu;
};

#endif //NEIGHBOR_GRID_CUH