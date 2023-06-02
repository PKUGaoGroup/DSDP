#ifndef COPY_MOL2_FORMAT_CUH
#define COPY_MOL2_FORMAT_CUH
#include "common.cuh"

//通过输入的pdbqt文件和mol2文件进行比较（需要保证pdbqt坐标和mol2匹配）
//能将得到的坐标文件以一样的mol2格式保存
struct COPY_pdbqt_FORMAT
{
	int last_modified_time = 20221109;

	int atom_numbers_in_pdbqt = 0;


	std::vector<VECTOR>crd_in_pdbqt;


	//实际按数组形式维护
	int pdbqt_content_numbers = -1;
	std::vector<int> pdbqt_atom_list;
	std::vector<std::string> pdbqt_content;//最大容纳512


	//std::vector<int>mol2_serial_map_to_pdbqt_serial;//长度为mol2的，每个mol2对应pdbqt中的一个原子，没有对应的则赋为-1；

	void Initial(char* pdbqt_name);
	//void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking);
	//在输出坐标的时候对每个原子都加一个move_vec的偏移矢量
	void Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec);
	
	//multi pdbqt in one file
	void Append_Frame_To_Opened_pdbqt_standard(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec,const int pose_rank,const float score);
};
#endif //COPY_MOL2_FORMAT_CUH
