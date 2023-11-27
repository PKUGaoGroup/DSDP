#ifndef COPY_MOL2_FORMAT_CUH
#define COPY_MOL2_FORMAT_CUH
#include "common.cuh"

struct COPY_pdbqt_FORMAT
{
	int atom_numbers_in_pdbqt = 0;

	std::vector<VECTOR>crd_in_pdbqt;

	//实际按数组形式维护
	int pdbqt_content_numbers = -1;//Initial之前为-1。
	std::vector<int> pdbqt_atom_list;//记录每一行是否是ATOM||HETATM行
	std::vector<std::string> pdbqt_content;//check 最大容纳512

	//读入一个pdbqt作为初始化。
	//分子构象变化后可以相同的格式输出
	void Initial(char* pdbqt_name);
	
	//在输出坐标的时候对每个原子都加一个move_vec的偏移矢量
	void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec);

	//额外补充排名和打分
	void Append_Frame_To_Opened_pdbqt_standard(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec, const int pose_rank, const float score);
};
#endif //COPY_MOL2_FORMAT_CUH

