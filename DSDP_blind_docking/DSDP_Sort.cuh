#ifndef DSDP_SORT_CUH
#define DSDP_SORT_CUH
#include "common.cuh"

struct DSDP_SORT
{
	int last_modified_time = 20221114;

	int atom_numbers=0;
	int selected_numbers = 0;
	std::vector<float> selected_energy;//不要用size()等函数，desired_selecting_numbers不一定相等
	std::vector<VECTOR> selected_crd;
	//在标准DSDP的末尾，对能量排序后，使用
	//record_numbers是整个搜索过程存储下来的坐标crd_record的数目，也是按能量拍好序的序号-能量列表serial_energy_list的长度（出于效率考虑，一般可以只给个min(2000,record_numbers)）
	//rmsd_cutoff是两个构象相似与否的判据（一般为2.f），
	// 比较相似结构时，只向已被选出的最近forward_comparing_numbers个构象进行比较（一般为20，在计算效率上考虑）
	//希望排出前desired_selecting_numbers名用于后续打分（一般50，没理由全部排序）
	//
	void Sort_Structures
	(const int atom_numbers,const int *atomic_number,
		const int record_numbers, const VECTOR* crd_record, const INT_FLOAT* serial_energy_list,
		const float rmsd_cutoff, const int forward_comparing_numbers,const int desired_selecting_numbers);
};

#endif //DSDP_SORT_CUH