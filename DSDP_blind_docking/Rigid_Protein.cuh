#ifndef RIGID_PROTEIN_CUH
#define RIGID_PROTEIN_CUH
#include "common.cuh"

struct RIGID_PROTEIN
{
	int last_modified_time = 20221109;

	std::vector<VECTOR> crd;
	std::vector<float> charge;
	VECTOR protein_center;
	std::vector<int> atomic_number;
	std::vector<int> atom_type;
	int atom_numbers = 0;
	VECTOR skin = {30.f,30.f,30.f};//让蛋白每个原子都不靠近模拟空间边界的skin范围内
	VECTOR protein_size;//蛋白质的长宽高（加上了skin）
	VECTOR move_vec;//原始的坐标信息加上move_vec得到程序中实际的坐标

	//vina_atom
	std::vector<VINA_ATOM> vina_atom;
	VINA_ATOM* d_vina_atom = NULL;
	int malloced_atom_numbers = 0;

	void Initial_Protein_From_PDBQT(const char* file_name,const VECTOR box_length);
private:
	VECTOR Find_A_Proper_Box();
};
#endif //RIGID_PROTEIN_CUH
