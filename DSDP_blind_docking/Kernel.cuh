#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "common.cuh"

#define HYDROGEN_ATOM_TYPE_SERIAL 17
#define MAX_OPTIMIZE_STEPS 100
#define BIG_ENERGY 1000.f
#define CONVERGENCE_CUTOFF 0.0001f


//inner_interaction_list是这样一种结构，用于记录每个原子需要考虑计算同一分子内两体作用的列表
//为方便分配，实际上inner_interaction_list是个atom_numbers*atom_numbers的矩阵
//但每个inner_interaction_list[i*atom_numbers]代表i号原子要考虑两体作用的原子数（存储的考虑编号总是大于i）
//且为了保证效率，要求每一行inner_interaction_list[i*atom_numbers]后面的原子序号都是排了序的。
//frc和energy都会在该kernel里清零重加，因此无需保证输入的frc和energy初始化
//为保持一致性，原子crd坐标均采用VECTOR_INT，int记录的是原子种类。
__global__ void Calculate_Energy_And_Grad_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse
);

//算力相关的变量可参考上面的kernel函数
//ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
//atom_to_node_serial
__global__ void Optimize_Structure_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
);


//算力相关的变量可参考上面的kernel函数
//ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
//atom_to_node_serial
__global__ void Optimize_Structure_BB2_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
);


//对pair作用不使用插值表，直接进行计算
//经测试，比上面的使用pair作用插值的要快很多，重点并不是在于减少计算量（反而增加了极大的计算量），而是在于避免每个kernel都读取公共的pair_potential
__global__ void Optimize_Structure_BB2_Direct_Pair_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VINA_ATOM* vina_atom, VECTOR* frc, float* energy,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
);
#endif //KERNEL_CUH
