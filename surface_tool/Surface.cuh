#ifndef SURFACE_CUH
#define SURFACE_CUH

//实际对应放置Rigid_Protein.cuh的位置，实际编译时也需要连接common.o，为方便可以复制一份放在一起
#include "common.cuh"

struct SURFACE
{
	float skin = 3.f;//对整个蛋白各方向向外skin长度，保证卷积操作和原子半径不出边界
	VECTOR move_vec;//相对读入的原子坐标的平移矢量（可参考Rigid_Protein.cuh）

	std::vector<float>atom_radius;//由Initial函数自带参数
	std::vector<VECTOR>atom_crd;
	std::vector<int>atomic_number;
	std::vector<int>atom_is_near_surface;//最终存储的结果，1则说明是在边界，0则不是

	//GPU上相关
	VECTOR* d_atom_crd = NULL;
	float* d_atom_radius = NULL;
	int* d_atom_is_near_surface = NULL;

	int extending_numbers;//这个和skin以及原子半径相关（越大计算越慢，但在GPU上单次运行不明显，可以较大）
						//主要用于保证每个原子能遍历到其半径范围内所覆盖到的格点
	//网格的基本参数，要求格子是立方体，所以grid_length等是float而非VECTOR
	INT_VECTOR grid_dimension;
	INT_VECTOR grid_dimension_minus_one;
	int layer_numbers;
	int grid_numbers;
	float grid_length_inverse;
	float grid_length;
	VECTOR box_length;//一般情况下尽可能刚好比蛋白大一点（由skin和蛋白尺寸决定）

	int* d_origin_grid_occupation = NULL;//0，1三维网格，记录原始的蛋白占据情况
	int* d_smoothed_grid_occupation = NULL;

	//一个函数直接出结果，注意没有free()操作，因此这个结构体实际上只允许运行一次，如有需要可进一步修改
	void Initial(const float grid_length, const int atom_numbers, const VECTOR* atom_crd, const int* atomic_number);
};

#endif //SURFACE_CUH
