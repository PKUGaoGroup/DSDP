#ifndef VINA_GRID_FORCE_FIELD_CUH
#define VINA_GRID_FORCE_FIELD_CUH
#include "common.cuh"

//�������ײ�ֵ����
struct VINA_GRID_FORCE_FIELD
{
	//ֻ������������ӣ���Ҫ��֤������������Ǵ����������򼴿�
	int type_numbers = 18;//�ܹ�18��ԭ�����ͣ�����Ҫ18��grid���洢��ͬԭ����ͬһ��λ�ӿ��ܸ��ܵ�������
	std::vector<VINA_ATOM> basic_vina_atom;
	VINA_ATOM* d_basic_vina_atom = NULL;

	float cutoff = 8.f;
	VECTOR grid_min;//��ֵ���ӵ���С����ʵ�ʿռ��е�λ�ã���һ�������������е�box_min��Ӧ��Ӧ�ø������Ա�֤���Ӳ����߳����grid���������������ʱ�ܲ����뿪���ӣ�
	VECTOR box_length;//������ֵ�ռ����ά�ߴ磨ʵ����������Ӧ��һ����
	VECTOR grid_length;//��Ԫ����ĳߴ�
	VECTOR grid_length_inverse;
	INT_VECTOR grid_dimension;//�������Ŀ��������������������Ա߽���������ʵ�ʸ�����Ҫ�Ȳ�ֵ����һ����ÿ����
	int layer_numbers = 0;
	int grid_numbers = 0;//ʵ���Ǹ���������Ǹ���������������Ϊa*a*a���������ʵ��ֻ��(a-1)*(a-1)*(a-1)

	//�����������ռ�ĵ��߻��ָ�����ʹ��ʹ�õĽضϰ뾶���г�ʼ��
	//������Ҫ��ÿ����ϵÿ��λ����£����initialֻ����������ڴ����Ȳ���
	//grid_numbers_in_one_dimemsion��������ÿ�������ϵĲ�ֵ�����Ŀ
	void Initial(const int grid_numbers_in_one_dimemsion, const float cutoff);

	struct GRID
	{
		VINA_GRID_FORCE_FIELD* vina_grid_force_field = NULL;
		float4* potential = NULL;//Ŀǰʹ�����������ֱ��ֵ�ķ���

		//GPU��texture�ڴ���ر���
		long long int* texObj_for_kernel = NULL;//ʵ��Ϊ����ָ�룬ָ��18�����͵Ĳ�ֵ�����ַtexObj_potential
		std::vector < cudaArray* > cudaArray_potential;//��ֵ���ʵ�ʴ洢��λ��
		std::vector<cudaMemcpy3DParms> copyParams_potential;//�ڴ渴��ʱʹ�õĲ���
		std::vector<cudaTextureObject_t> texObj_potential;//ʵ���Ǹ�long long int�������ʵ��kernel������ʹ��ʱ����������һ��texture�ı������

		//�������������ռ��ʵ��������С��box_min�������ռ���ӱ߳�box_length��
		//������ԭ����atom_numbers������ԭ����Ϣd_protein����GPU�ϵĵ���ԭ����Ϣ�����ꡢ���ࡢ�뾶...����
		//����ԭ������Ŀռ仮����neighbor_grid_bucket�д洢
		//���е��������ֵ�����ļ���
		void Calculate_Protein_Potential_Grid
		(
			const VECTOR box_min, const float box_length,
			const int atom_numbers, const VINA_ATOM* d_protein,
			const VECTOR neighbor_grid_length_inverse, const INT_VECTOR neighbor_grid_dimension, const NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket
		);

		void Cuda_Texture_Initial(VINA_GRID_FORCE_FIELD* vgff);//��ʼ��texture�ڴ�
	private:
		void Copy_Potential_To_Texture();
	}grid;
};

#endif //VINA_GRID_FORCE_FIELD_CUH