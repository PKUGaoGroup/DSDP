#ifndef PARTIAL_RIGID_SMALL_MOLECULE_CUH
#define PARTIAL_RIGID_SMALL_MOLECULE_CUH
#include "common.cuh"


//主要结构体，考虑到实际需要，同时保留极性氢显式力场（进一步适配需要）和隐式力场（适配vina打分函数）的可能
//为保证高效，在多stream操纵时，单个结构体从pdbqt文件初始化，其他则直接复制结构体
//为保证高效，在读入pdbqt文件时，同时生成显示极性氢的tree与隐式极性氢的tree
//为方便，原子排序保持和pdbqt一致，由于pdbqt自动满足tree的顺序结构，因此没必要换序
struct PARTIAL_RIGID_SMALL_MOLECULE
{
	int last_modified_time = 20221110;

	//分子的基本信息
	int atom_numbers = 0;
	std::vector<VECTOR>frc;//原子上的力
	std::vector<VECTOR>crd;//原子的坐标
	std::vector<VECTOR>crd_from_pdbqt;//读入pdbqt中原子的坐标
	std::vector<VECTOR>origin_crd;//首号原子坐标为0时的坐标，用于生成crd
	void Refresh_origin_crd(const VECTOR * new_crd); // be used to refresh origin crd
	//currently, it only refreshes data on CPU. Be careful!
	VECTOR move_vec;//记录origin_crd相对原始读入的pdbqt的位移矢量（用于与初始pdbqt构象做比较），实际为原始的crd[0]位置。
	std::vector<int>atomic_number;//原子序数
	std::vector<int>atom_mass;//原子质量，暂时没有作用
	std::vector<float>charge;//原子电荷，暂时没有作用
	std::vector<int>atom_type;//原子种类的描述，实际上是个char[4]，对于pdbqt来说，实际就是第77、78列的str\

	//二面角自由度的拓扑结构存储
	struct TREE
	{
		int torsion_numbers = 0;//有效二面角数量，与常见的node_numbers等同
		std::vector<NODE>node;
		std::vector<int>atom_to_node_serial;//每个原子到所属node的序号（pdbqt中root内的原子的node为-1，只有整体转动和平动自由度）
	};

	TREE pdbqt_tree;
	std::vector<int> is_pure_H_freedom;//记录pdbqt tree中每个节点是否对应一个纯氢的转动（纯氢指羟基等集团，在vina中不存在转动自由度）
										//为方便构建vina tree，这个信息同时也记录了pdbqt原始的原子列表分片节点信息
	TREE vina_tree;

	float num_tor = 0.f;//vina打分使用的参数，对应为二面角自由度总数（但氢自由度算0.5）

	void Initial_From_PDBQT(const char* file_name);
	void Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(PARTIAL_RIGID_SMALL_MOLECULE* input);

	//gpu相关
	struct GPU
	{
		PARTIAL_RIGID_SMALL_MOLECULE* partial_rigid_small_molecule = NULL;
		int u_freedom = 0;//由于vina和pdbqt的实际自由度不同，因此总自由度放在gpu内部
		int atom_numbers = 0;//和父结构体同样的信息，方便调用
		int node_numbers = 0;

		int malloced_atom_numbers = 0;
		VECTOR* origin_crd = NULL;//一般是root原子坐标平移到(0,0,0)的最原始输入坐标，在整个搜索过程中保持不变（放置数值导致“嵌入空间变形”）
		VECTOR* ref_crd = NULL;//实际生成crd时使用的参考坐标，该坐标与origin_crd相比，一般只差一个整体转动矩阵。
		VECTOR* crd = NULL;
		VECTOR* last_crd = NULL;//近邻表更新判据
		VECTOR* frc = NULL;
		int* atom_to_node_serial = NULL;
		int* inner_neighbor_list = NULL;//小分子内部的原子间相互作用近邻表，由于小分子的普遍大小，实际就相当于两两进行一次计算，但由于需要剔除一些原子对（1-2，1-3，1-4作用），因此直接搞成一个固定的近邻表，所以其实际大小为atom_numbers*atom_numbers。实际应该用三角阵，但目前用完整的
		int* h_inner_neighbor_list = NULL;

		int malloced_u_freedom = 0;
		float* h_u_crd = NULL;//方便外界初始化自由度坐标以及偶尔进行大幅度扰动强制接受
		float* u_crd = NULL;
		float* last_u_crd = NULL;//BB opt
		float* h_last_accepted_u_crd = NULL;//MC
		float* dU_du_crd = NULL;
		float* last_dU_du_crd = NULL;//BB opt

		int malloced_node_numbers = 0;
		NODE* node = NULL;

		void Initial(PARTIAL_RIGID_SMALL_MOLECULE* mol, const TREE* tree);
	

		//MC和vina力场相关
		float last_accepted_energy = 100.f;
		VINA_ATOM* h_vina_atom = NULL;
		VINA_ATOM* d_vina_atom = NULL;
	};

	//目前只暂时考虑使用vina力场的过程
	GPU vina_gpu;
	GPU pdbqt_gpu;//如果考虑有H的自由度参与
	
	
	//the neighbor list is prepared for structure refine correctly;
	const int MAX_NEIGHBOR_NUMBERS=1024;
	int *neighbor_list_total=NULL;
	std::vector<int*>neighbor_list;//the 0-th element is the number of neighbors
	
	//compute PARTIAL_RIGID_SMALL_MOLECULE::crd & protein_crd neighbor_list
	void Build_Neighbor_List(const float cutoff_with_skin,const int protein_atom_numbers, const VECTOR *protein_crd, const int * protein_atomic_number);
	float Refine_Structure(const VINA_ATOM *protein,const VECTOR*protein_crd);
	std::vector<float>dU_du_crd;
	std::vector<float>u_crd;
	float* energy= NULL;
};
#endif //PARTIAL_RIGID_SMALL_MOLECULE_CUH
