#ifndef RIGID_PROTEIN_CUH
#define RIGID_PROTEIN_CUH
#include "common.cuh"

struct RIGID_PROTEIN
{
	int last_modified_time = 20221109;

	std::vector<VECTOR> crd;
	std::vector<float> charge;
	std::vector<int> atomic_number;
	std::vector<int> atom_type;
	int atom_numbers = 0;
	VECTOR skin = {30.f,30.f,30.f};//�õ���ÿ��ԭ�Ӷ�������ģ��ռ�߽��skin��Χ��
	VECTOR protein_size;//�����ʵĳ����ߣ�������skin��
	VECTOR move_vec;//ԭʼ��������Ϣ����move_vec�õ�������ʵ�ʵ�����

	//vina_atom
	std::vector<VINA_ATOM> vina_atom;
	VINA_ATOM* d_vina_atom = NULL;
	int malloced_atom_numbers = 0;

	void Initial_Protein_From_PDBQT(const char* file_name,const VECTOR box_length);
private:
	VECTOR Find_A_Proper_Box();
};
#endif //RIGID_PROTEIN_CUH