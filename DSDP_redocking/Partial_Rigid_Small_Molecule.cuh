#ifndef PARTIAL_RIGID_SMALL_MOLECULE_CUH
#define PARTIAL_RIGID_SMALL_MOLECULE_CUH
#include "common.cuh"


//��Ҫ�ṹ�壬���ǵ�ʵ����Ҫ��ͬʱ������������ʽ��������һ��������Ҫ������ʽ����������vina��ֺ������Ŀ���
//Ϊ��֤��Ч���ڶ�stream����ʱ�������ṹ���pdbqt�ļ���ʼ����������ֱ�Ӹ��ƽṹ��
//Ϊ��֤��Ч���ڶ���pdbqt�ļ�ʱ��ͬʱ������ʾ�������tree����ʽ�������tree
//Ϊ���㣬ԭ�����򱣳ֺ�pdbqtһ�£�����pdbqt�Զ�����tree��˳��ṹ�����û��Ҫ����
struct PARTIAL_RIGID_SMALL_MOLECULE
{
	int last_modified_time = 20221110;

	//���ӵĻ�����Ϣ
	int atom_numbers = 0;
	std::vector<VECTOR>frc;//ԭ���ϵ���
	std::vector<VECTOR>crd;//ԭ�ӵ�����
	std::vector<VECTOR>crd_from_pdbqt;//����pdbqt��ԭ�ӵ�����
	std::vector<VECTOR>origin_crd;//�׺�ԭ������Ϊ0ʱ�����꣬��������crd
	void Refresh_origin_crd(const VECTOR * new_crd); // be used to refresh origin crd
	//currently, it only refreshes data on CPU. Be careful!



	VECTOR move_vec;//��¼origin_crd���ԭʼ�����pdbqt��λ��ʸ�����������ʼpdbqt�������Ƚϣ���ʵ��Ϊԭʼ��crd[0]λ�á�
	std::vector<int>atomic_number;//ԭ������
	std::vector<int>atom_mass;//ԭ����������ʱû������
	std::vector<float>charge;//ԭ�ӵ�ɣ���ʱû������
	std::vector<int>atom_type;//ԭ�������������ʵ�����Ǹ�char[4]������pdbqt��˵��ʵ�ʾ��ǵ�77��78�е�str\

	//��������ɶȵ����˽ṹ�洢
	struct TREE
	{
		int torsion_numbers = 0;//��Ч������������볣����node_numbers��ͬ
		std::vector<NODE>node;
		std::vector<int>atom_to_node_serial;//ÿ��ԭ�ӵ�����node����ţ�pdbqt��root�ڵ�ԭ�ӵ�nodeΪ-1��ֻ������ת����ƽ�����ɶȣ�
	};

	TREE pdbqt_tree;
	std::vector<int> is_pure_H_freedom;//��¼pdbqt tree��ÿ���ڵ��Ƿ��Ӧһ�������ת��������ָ�ǻ��ȼ��ţ���vina�в�����ת�����ɶȣ�
										//Ϊ���㹹��vina tree�������ϢͬʱҲ��¼��pdbqtԭʼ��ԭ���б���Ƭ�ڵ���Ϣ
	TREE vina_tree;

	float num_tor = 0.f;//vina���ʹ�õĲ�������ӦΪ��������ɶ��������������ɶ���0.5��

	void Initial_From_PDBQT(const char* file_name);
	void Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(PARTIAL_RIGID_SMALL_MOLECULE* input);

	//gpu���
	struct GPU
	{
		PARTIAL_RIGID_SMALL_MOLECULE* partial_rigid_small_molecule = NULL;
		int u_freedom = 0;//����vina��pdbqt��ʵ�����ɶȲ�ͬ����������ɶȷ���gpu�ڲ�
		int atom_numbers = 0;//�͸��ṹ��ͬ������Ϣ���������
		int node_numbers = 0;

		int malloced_atom_numbers = 0;
		VECTOR* origin_crd = NULL;//һ����rootԭ������ƽ�Ƶ�(0,0,0)����ԭʼ�������꣬���������������б��ֲ��䣨������ֵ���¡�Ƕ��ռ���Ρ���
		VECTOR* ref_crd = NULL;//ʵ������crdʱʹ�õĲο����꣬��������origin_crd��ȣ�һ��ֻ��һ������ת������
		VECTOR* crd = NULL;
		VECTOR* last_crd = NULL;//���ڱ������о�
		VECTOR* frc = NULL;
		int* atom_to_node_serial = NULL;
		int* inner_neighbor_list = NULL;//С�����ڲ���ԭ�Ӽ��໥���ý��ڱ�������С���ӵ��ձ��С��ʵ�ʾ��൱����������һ�μ��㣬��������Ҫ�޳�һЩԭ�Ӷԣ�1-2��1-3��1-4���ã������ֱ�Ӹ��һ���̶��Ľ��ڱ���������ʵ�ʴ�СΪatom_numbers*atom_numbers��ʵ��Ӧ���������󣬵�Ŀǰ��������
		int* h_inner_neighbor_list = NULL;

		int malloced_u_freedom = 0;
		float* h_u_crd = NULL;//��������ʼ�����ɶ������Լ�ż�����д�����Ŷ�ǿ�ƽ���
		float* u_crd = NULL;
		float* last_u_crd = NULL;//BB opt
		float* h_last_accepted_u_crd = NULL;//MC
		float* dU_du_crd = NULL;
		float* last_dU_du_crd = NULL;//BB opt

		int malloced_node_numbers = 0;
		NODE* node = NULL;

		void Initial(PARTIAL_RIGID_SMALL_MOLECULE* mol, const TREE* tree);
	

		//MC��vina�������
		float last_accepted_energy = 100.f;
		VINA_ATOM* h_vina_atom = NULL;
		VINA_ATOM* d_vina_atom = NULL;
	};

	//Ŀǰֻ��ʱ����ʹ��vina�����Ĺ���
	GPU vina_gpu;
	GPU pdbqt_gpu;//���������H�����ɶȲ���


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