#ifndef SURFACE_CUH
#define SURFACE_CUH

//ʵ�ʶ�Ӧ����Rigid_Protein.cuh��λ�ã�ʵ�ʱ���ʱҲ��Ҫ����common.o��Ϊ������Ը���һ�ݷ���һ��
#include "common.cuh"

struct SURFACE
{
	float skin = 3.f;//���������׸���������skin���ȣ���֤���������ԭ�Ӱ뾶�����߽�
	VECTOR move_vec;//��Զ����ԭ�������ƽ��ʸ�����ɲο�Rigid_Protein.cuh��

	std::vector<float>atom_radius;//��Initial�����Դ�����
	std::vector<VECTOR>atom_crd;
	std::vector<int>atomic_number;
	std::vector<int>atom_is_near_surface;//���մ洢�Ľ����1��˵�����ڱ߽磬0����

	//GPU�����
	VECTOR* d_atom_crd = NULL;
	float* d_atom_radius = NULL;
	int* d_atom_is_near_surface = NULL;

	int extending_numbers;//�����skin�Լ�ԭ�Ӱ뾶��أ�Խ�����Խ��������GPU�ϵ������в����ԣ����Խϴ�
						//��Ҫ���ڱ�֤ÿ��ԭ���ܱ�������뾶��Χ�������ǵ��ĸ��
	//����Ļ���������Ҫ������������壬����grid_length����float����VECTOR
	INT_VECTOR grid_dimension;
	INT_VECTOR grid_dimension_minus_one;
	int layer_numbers;
	int grid_numbers;
	float grid_length_inverse;
	float grid_length;
	VECTOR box_length;//һ������¾����ܸպñȵ��״�һ�㣨��skin�͵��׳ߴ������

	int* d_origin_grid_occupation = NULL;//0��1��ά���񣬼�¼ԭʼ�ĵ���ռ�����
	int* d_smoothed_grid_occupation = NULL;

	//һ������ֱ�ӳ������ע��û��free()�������������ṹ��ʵ����ֻ��������һ�Σ�������Ҫ�ɽ�һ���޸�
	void Initial(const float grid_length, const int atom_numbers, const VECTOR* atom_crd, const int* atomic_number);
};

#endif //SURFACE_CUH
