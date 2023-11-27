#ifndef RIGID_PROTEIN_CUH
#define RIGID_PROTEIN_CUH
#include "common.cuh"

struct RIGID_PROTEIN
{
	std::vector<VECTOR> crd;
	std::vector<float> charge;
	VECTOR protein_center;
	std::vector<int> atomic_number;
	std::vector<int> atom_type;
	int atom_numbers = 0;
	VECTOR skin = { 30.f,30.f,30.f };//在蛋白的尺寸基础上，额外加上一定的空间冗余。
	VECTOR protein_size;//蛋白的尺寸+2.f*skin
	VECTOR move_vec;//原始PDBQT中的坐标+move_vec=程序内的蛋白原子坐标

	//vina_atom
	std::vector<VINA_ATOM> vina_atom;
	VINA_ATOM* d_vina_atom = NULL;
	int malloced_atom_numbers = 0;

	//初始化蛋白质信息，需要给定一个允许的蛋白空间盒子大小。
	void Initial_Protein_From_PDBQT(const char* file_name, const VECTOR box_length);
private:
	//计算蛋白尺寸，用合适的盒子大小+平移矢量来包住蛋白
	VECTOR Find_A_Proper_Box();
};
#endif //RIGID_PROTEIN_CUH
