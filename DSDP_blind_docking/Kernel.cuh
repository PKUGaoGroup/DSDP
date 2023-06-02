#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "common.cuh"

#define HYDROGEN_ATOM_TYPE_SERIAL 17
#define MAX_OPTIMIZE_STEPS 100
#define BIG_ENERGY 1000.f
#define CONVERGENCE_CUTOFF 0.0001f


//inner_interaction_list������һ�ֽṹ�����ڼ�¼ÿ��ԭ����Ҫ���Ǽ���ͬһ�������������õ��б�
//Ϊ������䣬ʵ����inner_interaction_list�Ǹ�atom_numbers*atom_numbers�ľ���
//��ÿ��inner_interaction_list[i*atom_numbers]����i��ԭ��Ҫ�����������õ�ԭ�������洢�Ŀ��Ǳ�����Ǵ���i��
//��Ϊ�˱�֤Ч�ʣ�Ҫ��ÿһ��inner_interaction_list[i*atom_numbers]�����ԭ����Ŷ���������ġ�
//frc��energy�����ڸ�kernel�������ؼӣ�������豣֤�����frc��energy��ʼ��
//Ϊ����һ���ԣ�ԭ��crd���������VECTOR_INT��int��¼����ԭ�����ࡣ
__global__ void Calculate_Energy_And_Grad_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse
);

//������صı����ɲο������kernel����
//ref_crd�Ƕ�Ӧu_crd��node�Ĳο����꣬����vina_atom�ڵ��������������
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


//������صı����ɲο������kernel����
//ref_crd�Ƕ�Ӧu_crd��node�Ĳο����꣬����vina_atom�ڵ��������������
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


//��pair���ò�ʹ�ò�ֵ��ֱ�ӽ��м���
//�����ԣ��������ʹ��pair���ò�ֵ��Ҫ��ܶ࣬�ص㲢�������ڼ��ټ����������������˼���ļ����������������ڱ���ÿ��kernel����ȡ������pair_potential
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
