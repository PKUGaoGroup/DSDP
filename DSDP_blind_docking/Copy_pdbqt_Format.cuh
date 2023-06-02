#ifndef COPY_MOL2_FORMAT_CUH
#define COPY_MOL2_FORMAT_CUH
#include "common.cuh"

//ͨ�������pdbqt�ļ���mol2�ļ����бȽϣ���Ҫ��֤pdbqt�����mol2ƥ�䣩
//�ܽ��õ��������ļ���һ����mol2��ʽ����
struct COPY_pdbqt_FORMAT
{
	int last_modified_time = 20221109;

	int atom_numbers_in_pdbqt = 0;


	std::vector<VECTOR>crd_in_pdbqt;


	//ʵ�ʰ�������ʽά��
	int pdbqt_content_numbers = -1;
	std::vector<int> pdbqt_atom_list;
	std::vector<std::string> pdbqt_content;//�������512


	//std::vector<int>mol2_serial_map_to_pdbqt_serial;//����Ϊmol2�ģ�ÿ��mol2��Ӧpdbqt�е�һ��ԭ�ӣ�û�ж�Ӧ����Ϊ-1��

	void Initial(char* pdbqt_name);
	//void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking);
	//����������ʱ���ÿ��ԭ�Ӷ���һ��move_vec��ƫ��ʸ��
	void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec);
	
	//multi pdbqt in one file
	void Append_Frame_To_Opened_pdbqt_standard(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec,const int pose_rank,const float score);
};
#endif //COPY_MOL2_FORMAT_CUH
