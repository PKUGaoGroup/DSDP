#ifndef DSDP_SORT_CUH
#define DSDP_SORT_CUH
#include "common.cuh"

struct DSDP_SORT
{
	int last_modified_time = 20221114;

	int atom_numbers=0;
	int selected_numbers = 0;
	std::vector<float> selected_energy;//��Ҫ��size()�Ⱥ�����desired_selecting_numbers��һ�����
	std::vector<VECTOR> selected_crd;
	//�ڱ�׼DSDP��ĩβ�������������ʹ��
	//record_numbers�������������̴洢����������crd_record����Ŀ��Ҳ�ǰ������ĺ�������-�����б�serial_energy_list�ĳ��ȣ�����Ч�ʿ��ǣ�һ�����ֻ����min(2000,record_numbers)��
	//rmsd_cutoff�������������������оݣ�һ��Ϊ2.f����
	// �Ƚ����ƽṹʱ��ֻ���ѱ�ѡ�������forward_comparing_numbers��������бȽϣ�һ��Ϊ20���ڼ���Ч���Ͽ��ǣ�
	//ϣ���ų�ǰdesired_selecting_numbers�����ں�����֣�һ��50��û����ȫ������
	//
	void Sort_Structures
	(const int atom_numbers,const int *atomic_number,
		const int record_numbers, const VECTOR* crd_record, const INT_FLOAT* serial_energy_list,
		const float rmsd_cutoff, const int forward_comparing_numbers,const int desired_selecting_numbers);
};

#endif //DSDP_SORT_CUH