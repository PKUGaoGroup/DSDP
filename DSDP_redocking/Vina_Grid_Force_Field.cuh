#ifndef VINA_GRID_FORCE_FIELD_CUH
#define VINA_GRID_FORCE_FIELD_CUH
#include "common.cuh"

//构建蛋白插值网格
struct VINA_GRID_FORCE_FIELD
{
	//只考虑立方体盒子，主要保证立方体盒子总是大于受限区域即可
	int type_numbers = 18;//总共18种原子类型，即需要18套grid来存储不同原子在同一个位子可能感受到的能量
	std::vector<VINA_ATOM> basic_vina_atom;
	VINA_ATOM* d_basic_vina_atom = NULL;

	float cutoff = 8.f;
	VECTOR grid_min;//插值格子的最小点在实际空间中的位置，不一定与搜索程序中的box_min对应，应该更宽泛，以保证分子不会走出这个grid的区域（理论上随机时能部分离开盒子）
	VECTOR box_length;//完整插值空间的三维尺寸（实际三个分量应该一样）
	VECTOR grid_length;//单元网格的尺寸
	VECTOR grid_length_inverse;
	INT_VECTOR grid_dimension;//网格的数目，由于这个不再是周期性边界的网格，因此实际格子数要比插值点少一个（每方向）
	int layer_numbers = 0;
	int grid_numbers = 0;//实际是格点数，而非格子数：如果格点数为a*a*a，则格子数实际只有(a-1)*(a-1)*(a-1)

	//以受限搜索空间的单边划分格点数和打分使用的截断半径进行初始化
	//由于需要对每个体系每个位点更新，这个initial只负责基本的内存分配等操作
	//grid_numbers_in_one_dimemsion即给定了每个方向上的插值格点数目
	void Initial(const int grid_numbers_in_one_dimemsion, const float cutoff);

	struct GRID
	{
		VINA_GRID_FORCE_FIELD* vina_grid_force_field = NULL;
		float4* potential = NULL;//目前使用能量和力分别插值的方案

		//GPU上texture内存相关变量
		long long int* texObj_for_kernel = NULL;//实际为二重指针，指向18个类型的插值网格地址texObj_potential
		std::vector < cudaArray* > cudaArray_potential;//插值格点实际存储的位置
		std::vector<cudaMemcpy3DParms> copyParams_potential;//内存复制时使用的参数
		std::vector<cudaTextureObject_t> texObj_potential;//实际是个long long int，因此在实际kernel函数中使用时，单独建立一个texture的编号数组

		//给定受限搜索空间的实际坐标最小点box_min，搜索空间盒子边长box_length，
		//蛋白总原子数atom_numbers，蛋白原子信息d_protein（在GPU上的蛋白原子信息：坐标、种类、半径...），
		//蛋白原子坐标的空间划分在neighbor_grid_bucket中存储
		//进行蛋白网格插值力场的计算
		void Calculate_Protein_Potential_Grid
		(
			const VECTOR box_min, const float box_length,
			const int atom_numbers, const VINA_ATOM* d_protein,
			const VECTOR neighbor_grid_length_inverse, const INT_VECTOR neighbor_grid_dimension, const NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket
		);

		void Cuda_Texture_Initial(VINA_GRID_FORCE_FIELD* vgff);//初始化texture内存
	private:
		void Copy_Potential_To_Texture();
	}grid;
};

#endif //VINA_GRID_FORCE_FIELD_CUH