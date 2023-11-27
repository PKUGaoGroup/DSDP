#ifndef DSDP_SORT_CUH
#define DSDP_SORT_CUH
#include "common.cuh"

//用于对给定一组构象和对应的能量进行去重
struct DSDP_SORT
{
	int atom_numbers = 0;//单个分子构象内含的原子数目
	int selected_numbers = 0;//被能量排序和RMSD去重后剩下的分子结构数
	std::vector<float> selected_energy;//不要用size()等函数，desired_selecting_numbers不一定相等，最终只有前selected_numbers个有意义
	std::vector<VECTOR> selected_crd;//被选择出来的构象，每atom_numbers个VECTOR一组依次排列


	//
	// check debug（只搜索一步可能会报错）
	// 在标准DSDP的末尾，对能量排序后，使用
	//record_numbers是整个搜索过程存储下来的坐标crd_record的数目，也是按能量拍好序的序号-能量列表serial_energy_list的长度（出于效率考虑，一般可以只给个min(2000,record_numbers)）
	//rmsd_cutoff是两个构象相似与否的判据（一般为2.f），
	// 比较相似结构时，只向已被选出的最近forward_comparing_numbers个构象进行比较（一般为20，在计算效率上考虑）
	//希望排出前desired_selecting_numbers名用于后续打分（一般50，没理由全部排序）
	//
	void Sort_Structures
	(const int atom_numbers, const int* atomic_number,
		const int record_numbers, const VECTOR* crd_record, const INT_FLOAT* serial_energy_list,
		const float rmsd_cutoff, const int forward_comparing_numbers, const int desired_selecting_numbers);
};

#endif //DSDP_SORT_CUH