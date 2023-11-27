#ifndef PARTIAL_RIGID_SMALL_MOLECULE_CUH
#define PARTIAL_RIGID_SMALL_MOLECULE_CUH
#include "common.cuh"

//主要结构体，考虑到实际需要，同时保留极性氢的显式力场（pdbqt_tree）和隐式力场（适配vina打分函数,vina_tree）
//目前使用隐式力场
//为保证高效，在多stream操纵时，单个结构体从pdbqt文件初始化，其他则直接复制结构体
//为保证高效，在读入pdbqt文件时，同时生成显示极性氢的tree与隐式极性氢的tree
//为方便，原子排序保持和pdbqt一致，由于pdbqt自动满足tree的顺序结构，因此没必要换序
struct PARTIAL_RIGID_SMALL_MOLECULE
{
	int last_modified_time = 20221110;

	//小分子的基本信息
	int atom_numbers = 0;
	std::vector<int>atomic_number;
	std::vector<int>atom_mass;
	std::vector<float>charge;
	std::vector<int>atom_type;
	std::vector<VECTOR>frc;
	std::vector<VECTOR>crd;//当前分子构象（可由origin_crd+u_crd生成）
	std::vector<VECTOR>crd_from_pdbqt;//从pdbqt中获得的原始坐标
	std::vector<VECTOR>origin_crd;//用广义坐标描述分子构象时，使用的参考构象
									//可根据需要更新，并注意更新对应的node等参数
	void Refresh_origin_crd(const VECTOR* new_crd); 

	VECTOR move_vec;//check ��¼origin_crd���ԭʼ�����pdbqt��λ��ʸ�����������ʼpdbqt�������Ƚϣ���ʵ��Ϊԭʼ��crd[0]λ�á�


	//二面角自由度的拓扑结构存储
	struct TREE
	{
		int torsion_numbers = 0;//有效二面角数量，与常见的node_numbers等同
		std::vector<NODE>node;//NODE见定义
		std::vector<int>atom_to_node_serial;//每个原子到所属node的序号（pdbqt中root内的原子的node为-1，只有整体转动和平动自由度）
	};

	TREE pdbqt_tree;//含显式H的拓扑
	std::vector<int> is_pure_H_freedom;//记录pdbqt tree中每个节点是否对应一个纯氢的转动（纯氢指羟基等集团，在vina中不存在转动自由度）
	TREE vina_tree;

	float num_tor = 0.f;//vina打分使用的参数，对应为二面角自由度总数（但氢自由度算0.5）

	void Initial_From_PDBQT(const char* file_name);
	void Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(PARTIAL_RIGID_SMALL_MOLECULE* input);//注意复制信息的完整性

	//gpu相关
	struct GPU
	{
		PARTIAL_RIGID_SMALL_MOLECULE* partial_rigid_small_molecule = NULL;
		int u_freedom = 0;
		int atom_numbers = 0;
		int node_numbers = 0;

		int malloced_atom_numbers = 0;
		VECTOR* origin_crd = NULL;//check
		VECTOR* ref_crd = NULL;
		VECTOR* crd = NULL;
		VECTOR* last_crd = NULL;
		VECTOR* frc = NULL;
		int* atom_to_node_serial = NULL;
		int* inner_neighbor_list = NULL;
		int* h_inner_neighbor_list = NULL;

		int malloced_u_freedom = 0;
		float* h_u_crd = NULL;
		float* u_crd = NULL;
		float* last_u_crd = NULL;//BB opt
		float* h_last_accepted_u_crd = NULL;//MC
		float* dU_du_crd = NULL;
		float* last_dU_du_crd = NULL;//BB opt

		int malloced_node_numbers = 0;
		NODE* node = NULL;

		void Initial(PARTIAL_RIGID_SMALL_MOLECULE* mol, const TREE* tree);


		//MC相关
		float last_accepted_energy = 100.f;
		VINA_ATOM* h_vina_atom = NULL;
		VINA_ATOM* d_vina_atom = NULL;
	};

	//目前只暂时考虑使用vina力场的过程（vina_gpu）
	GPU vina_gpu;
	GPU pdbqt_gpu;


	//在进行无蛋白插值网格近似的单次优化过程中，进行快捷的ligand-protein近邻表计算
	const int MAX_NEIGHBOR_NUMBERS = 1024;//每个ligand原子所能存储的最大的蛋白原子邻居数
	int* neighbor_list_total = NULL;
	std::vector<int*>neighbor_list;//和ligand-ligand近邻表相同的用法

	void Build_Neighbor_List(const float cutoff_with_skin, const int protein_atom_numbers, const VECTOR* protein_crd, const int* protein_atomic_number);
	float Refine_Structure(const VINA_ATOM* protein, const VECTOR* protein_crd);
	std::vector<float>dU_du_crd;
	std::vector<float>u_crd;
};
#endif //PARTIAL_RIGID_SMALL_MOLECULE_CUH