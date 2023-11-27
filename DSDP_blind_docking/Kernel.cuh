#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "common.cuh"


#define HYDROGEN_ATOM_TYPE_SERIAL 17 //H元素的Vina atom中atom type
#define MAX_OPTIMIZE_STEPS 100 //每次优化跑100步
#define BIG_ENERGY 1000.f //初始默认打分
#define CONVERGENCE_CUTOFF 0.0001f //check

//使用BB方法优化分子结构（ligand-ligand作用按打分公式直接计算inner_interaction_list，ligand-protein作用用插值网格计算protein_mesh）
//在参考构象ref_crd基础上，使用广义坐标u_crd描述分子构象（u_freedom=node_numbers个二面角+3个平动+3个转动）
//atom_to_node_serial为每个原子所属的节点编号
//frc为计算过程中暂存的原子上的受力（最终会被转化到梯度dU_du_crd上）
//box_border_strenth为插值网格边界墙的弹簧强度
//box_min，box_max为protein_mesh插值网格的位置（用两点确定一个长方体）
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
