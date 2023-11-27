#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "common.cuh"


#define HYDROGEN_ATOM_TYPE_SERIAL 17 //HԪ�ص�Vina atom��atom type
#define MAX_OPTIMIZE_STEPS 100 //ÿ���Ż���100��
#define BIG_ENERGY 1000.f //��ʼĬ�ϴ��
#define CONVERGENCE_CUTOFF 0.0001f //check

//ʹ��BB�����Ż����ӽṹ��ligand-ligand���ð���ֹ�ʽֱ�Ӽ���inner_interaction_list��ligand-protein�����ò�ֵ�������protein_mesh��
//�ڲο�����ref_crd�����ϣ�ʹ�ù�������u_crd�������ӹ���u_freedom=node_numbers�������+3��ƽ��+3��ת����
//atom_to_node_serialΪÿ��ԭ�������Ľڵ���
//frcΪ����������ݴ��ԭ���ϵ����������ջᱻת�����ݶ�dU_du_crd�ϣ�
//box_border_strenthΪ��ֵ����߽�ǽ�ĵ���ǿ��
//box_min��box_maxΪprotein_mesh��ֵ�����λ�ã�������ȷ��һ�������壩
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
