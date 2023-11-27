#ifndef COPY_MOL2_FORMAT_CUH
#define COPY_MOL2_FORMAT_CUH
#include "common.cuh"

struct COPY_pdbqt_FORMAT
{
	int atom_numbers_in_pdbqt = 0;

	std::vector<VECTOR>crd_in_pdbqt;

	//ʵ�ʰ�������ʽά��
	int pdbqt_content_numbers = -1;//Initial֮ǰΪ-1��
	std::vector<int> pdbqt_atom_list;//��¼ÿһ���Ƿ���ATOM||HETATM��
	std::vector<std::string> pdbqt_content;//check �������512

	//����һ��pdbqt��Ϊ��ʼ����
	//���ӹ���仯�������ͬ�ĸ�ʽ���
	void Initial(char* pdbqt_name);
	
	//����������ʱ���ÿ��ԭ�Ӷ���һ��move_vec��ƫ��ʸ��
	void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec);

	//���ⲹ�������ʹ��
	void Append_Frame_To_Opened_pdbqt_standard(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec, const int pose_rank, const float score);
};
#endif //COPY_MOL2_FORMAT_CUH

