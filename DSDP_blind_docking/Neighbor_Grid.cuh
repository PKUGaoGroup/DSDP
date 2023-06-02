#ifndef NEIGHBOR_GRID_CUH
#define NEIGHBOR_GRID_CUH
#include "common.cuh"



struct NEIGHBOR_GRID
{
	int last_modified_time = 20221109;

	NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket = NULL;
	INT_VECTOR grid_dimension;
	VECTOR box_length;
	VECTOR grid_length;
	VECTOR grid_length_inverse;
	float skin = 0.f;
	float skin2 = 0.f;
	float cutoff = 0.f;
	float cutoff2 = 0.f;
	int grid_numbers = 0;
	int layer_numbers = 0;

	//ֻ�����ʼ��һ�Σ����ܷ�����ʼ��������Ҫ��
	void Initial(VECTOR box_length, float cutoff, float skin);
	void Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd);
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