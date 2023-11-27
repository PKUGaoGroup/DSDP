#ifndef NEIGHBOR_GRID_CUH
#define NEIGHBOR_GRID_CUH
#include "common.cuh"

//对蛋白原子进行空间网格划分，在不同位置可通过所处格子获得蛋白近邻表
struct NEIGHBOR_GRID
{
	NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket = NULL;
	INT_VECTOR grid_dimension;
	VECTOR box_length;
	VECTOR grid_length;
	VECTOR grid_length_inverse;
	float skin = 0.f;//在cutoff的基础上外延一定距离
	float skin2 = 0.f;
	float cutoff = 0.f;
	float cutoff2 = 0.f;
	int grid_numbers = 0;//三维数组的总大小Nx*Ny*Nz
	int layer_numbers = 0;//一层的数目Nx*Ny

	//只允许初始化一次，不能反复初始化（合理要求）
	//要保证box_length足够大，能完全包住整个蛋白
	//注意与插值网格做区分，这里的盒子是建立这个蛋白网格近邻表所用
	void Initial(VECTOR box_length, float cutoff, float skin);

	//将每个蛋白原子放到其空间所属的网格中
	//之后，从网格中可直接提取附近蛋白原子的编号
	void Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd);
	
	//清空网格内的原子记录
	void Clear_Grid_Bucket_Total();
	struct GPU
	{
		NEIGHBOR_GRID* neighbor_grid = NULL;

		NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket = NULL;

		//在CPU上实现，并将计算结果传到GPU，因为这个操作对固定蛋白只进行一次
		void Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd);
	}gpu;
};

#endif //NEIGHBOR_GRID_CUH