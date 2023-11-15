#include "Kernel.cuh"
__device__ __host__ static void Matrix_Multiply_Vector(VECTOR* __restrict__ c, const float* __restrict__ a, const VECTOR* __restrict__ b)
{
	c[0].x = a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z;
	c[0].y = a[3] * b[0].x + a[4] * b[0].y + a[5] * b[0].z;
	c[0].z = a[6] * b[0].x + a[7] * b[0].y + a[8] * b[0].z;
}

//inner_interaction_list������һ�ֽṹ�����ڼ�¼ÿ��ԭ����Ҫ���Ǽ���ͬһ�������������õ��б�
//Ϊ������䣬ʵ����inner_interaction_list�Ǹ�atom_numbers*atom_numbers�ľ���
//��ÿ��inner_interaction_list[i*atom_numbers]����i��ԭ��Ҫ�����������õ�ԭ�������洢�Ŀ��Ǳ�����Ǵ���i��
//��Ϊ�˱�֤Ч�ʣ�Ҫ��ÿһ��inner_interaction_list[i*atom_numbers]�����ԭ����Ŷ���������ġ�
//frc��energy�����ڸ�kernel�������ؼӣ�������豣֤�����frc��energy��ʼ��
//Ϊ����һ���ԣ�ԭ��crd���������VECTOR_INT��int��¼����ԭ�����ࡣ
__global__ void Calculate_Energy_And_Grad_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse
)
{
	if (threadIdx.x == 0)
	{
		energy[0] = 0.f;
	}
	float total_energy_in_thread = 0.f;
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		VECTOR_INT atom_i = vina_atom[i];
		VECTOR force_i = { 0.f,0.f,0.f };
		VECTOR dr;
		if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//Ҫ���Ƿ���ԭ��
		{
			//box interaction
			dr.x = fdimf(box_min.x, atom_i.x);//��������ں����⣬���ṩһ������ʸ����ָ������ڷ���
			dr.y = fdimf(box_min.y, atom_i.y);
			dr.z = fdimf(box_min.z, atom_i.z);
			force_i.x += box_border_strenth * dr.x;
			force_i.y += box_border_strenth * dr.y;
			force_i.z += box_border_strenth * dr.z;
			total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

			dr.x = fdimf(atom_i.x, box_max.x);
			dr.y = fdimf(atom_i.y, box_max.y);
			dr.z = fdimf(atom_i.z, box_max.z);
			force_i.x -= box_border_strenth * dr.x;
			force_i.y -= box_border_strenth * dr.y;
			force_i.z -= box_border_strenth * dr.z;
			total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

			//protein interaction
			VECTOR serial;//�ڵ��ײ�ֵ�����еĸ������
			serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
			serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
			serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
			float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//�Զ���ֵ����Ҫƫ��������
			total_energy_in_thread += ans.w;
			force_i.x += ans.x;
			force_i.y += ans.y;
			force_i.z += ans.z;
		}
		frc[i] = force_i;//��kernel�����������frc���ۼ�
	}
	__syncthreads();//ͬ�����Ա�֤�����������ü���ʱ�Ѿ�ȫ���߹���һ����ͬʱ��֤����Ҳ�ɹ�����

	VECTOR_INT atom_i, atom_j;
	VECTOR force_i, temp_force;
	VECTOR dr;
	float rij, dd, dd_, frc_abs, rij_inverse;
	float4 ans;
	int inner_list_start;
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		atom_i = vina_atom[i];
		force_i = { 0.f,0.f,0.f };
		inner_list_start = i * atom_numbers;
		int inner_numbers = inner_interaction_list[inner_list_start];
		for (int k = 1; k <= inner_numbers; k = k + 1)
		{
			int j = inner_interaction_list[inner_list_start + k];
			atom_j = vina_atom[j];
			dr = { atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z };
			rij = norm3df(dr.x, dr.y, dr.z);//ʸ������
			if (rij < cutoff)
			{
				rij_inverse = 1.f / (rij + 10.e-6f);
				rij *= pair_potential_grid_length_inverse;//��Ϊ�������ò�ֵ���ĸ������
				dd = rij - floor(rij);
				dd_ = 1.f - dd;
				ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//���Դ���ֵ���ɿ��Ǻ����滻Ϊ���������һά��������protein meshһ��
				frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//�����־���ʸ������һ���ĶԽ�Ч������ã��������ڵ��������õ������ݾ���������������ϸ���ȷ�����������������Ǹ�˹˥���ģ�����ֻ�൱��ƽ����뱻��Զ��һ�㣩
				total_energy_in_thread += 0.5f * (dd_ * ans.y + dd * ans.w);//�����γ���2

				temp_force.x = frc_abs * dr.x;
				temp_force.y = frc_abs * dr.y;
				temp_force.z = frc_abs * dr.z;
				force_i.x += temp_force.x;
				force_i.y += temp_force.y;
				force_i.z += temp_force.z;
				atomicAdd(&frc[j].x, -temp_force.x);
				atomicAdd(&frc[j].y, -temp_force.y);
				atomicAdd(&frc[j].z, -temp_force.z);
			}
		}
		atomicAdd(&frc[i].x, force_i.x);
		atomicAdd(&frc[i].y, force_i.y);
		atomicAdd(&frc[i].z, force_i.z);
	}
	atomicAdd(&energy[0], total_energy_in_thread);
}

//������صı����ɲο������kernel����
//ref_crd�Ƕ�Ӧu_crd��node�Ĳο����꣬����vina_atom�ڵ��������������
//atom_to_node_serial
__global__ void Optimize_Structure_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
)
{
	//Ϊ���ǿ��ܵļ��٣�������С�ĸ�����Ϣ���ŵ�shared��
	//
	__shared__ float shared_data[19];
	float* rot_matrix = &shared_data[2];
	float* alpha = &shared_data[17];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//��ʱ������
		shared_data[1] = BIG_ENERGY;//��ʱ������
		//shared_data[2]...shared_data[10]//����ת������
		//shared_data[11] cos_b ����Ӧŷ��ת����
		//shared_data[12] sin_b
		//shared_data[13] cos_a
		//shared_data[14] sin_a
		//shared_data[15] cacb
		//shared_data[16] cbsa
		//shared_data[17] = 0.f;//BB�Ż��õ�alpha
		//shared_data[18] = 0.f;

	}

	//������ѭ��ǰ�Ļ�����ʼ��
	for (int i = 0; i < u_freedom; i = i + 1)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	//������ѭ��ǰ����ͬ��
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//�ڵ�ǰ���������¸��¸�ת������
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8];//cacb
				shared_data[16] = rot_matrix[5];//cbsa
			}
		}
		__syncthreads();

		//�ɸ�ת�������ԭʼ�������ɵ�ǰ����
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//����������frc�����ٺ���һ��ͬ��������
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//�����൱��Ҫ��node��a0��Ҫ��ref�����䣬��ѡ����ͬ��ԭ��
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//����ת���Ĳο�ԭ�����ǵ�һ��ԭ�ӣ�rootԭ�ӣ�
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//����ƽ����������
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//�ɵ�ǰ�������node��a��n���ڼ������������ʵ��˳��Ҳ����������ԭ��������У�
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = { vina_atom[node[node_id].root_atom_serial].x,vina_atom[node[node_id].root_atom_serial].y,vina_atom[node[node_id].root_atom_serial].z };
			tempn = { vina_atom[node[node_id].branch_atom_serial].x,vina_atom[node[node_id].branch_atom_serial].y,vina_atom[node[node_id].branch_atom_serial].z };
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//����ʵ�ʲ���Ҫͬ��

		//����ԭ������������
		float total_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR_INT atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VECTOR_INT atom_i = vina_atom[i];
			VECTOR force_i = { 0.f,0.f,0.f };
			VECTOR dr;
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//Ҫ���Ƿ���ԭ��
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.x);//��������ں����⣬���ṩһ������ʸ����ָ������ڷ���
				dr.y = fdimf(box_min.y, atom_i.y);
				dr.z = fdimf(box_min.z, atom_i.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.x, box_max.x);
				dr.y = fdimf(atom_i.y, box_max.y);
				dr.z = fdimf(atom_i.z, box_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				//protein interaction
				VECTOR serial;//�ڵ��ײ�ֵ�����еĸ������
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//�Զ���ֵ����Ҫƫ��������
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = { atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z };
				rij = norm3df(dr.x, dr.y, dr.z);//ʸ������
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse;//��Ϊ�������ò�ֵ���ĸ������
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//���Դ���ֵ���ɿ��Ǻ����滻Ϊ���������һά��������protein meshһ��
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//�����־���ʸ������һ���ĶԽ�Ч������ã��������ڵ��������õ������ݾ���������������ϸ���ȷ�����������������Ǹ�˹˥���ģ�����ֻ�൱��ƽ����뱻��Զ��һ�㣩
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w);//���inner list�ǲ��ظ�����pair���õģ�����Ҫ��0.5f

					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		__syncthreads();//�����Ӻ���ȫ�����ݶ��Լ�node�Ĳ�������Ϣ��ȫ

		//��ǰ�˳��Ż�
		if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		{
			if (threadIdx.x == 0)
			{
				energy[0] = shared_data[0];
			}
			break;
		}
		if (threadIdx.x == 0)
		{
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha[0] = 0.f;
			alpha[1] = 0.f;
		}

		//���������
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = { vina_atom[0].x ,vina_atom[0].y , vina_atom[0].z };
			VECTOR temp_crd2 = { vina_atom[i].x ,vina_atom[i].y , vina_atom[i].z };
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		//����BB�Ż�����(��ʱδ��������ת����ƽ���Ͷ�������ɶȵĸ����Ż�)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			atomicAdd(&alpha[0], y * s);
			atomicAdd(&alpha[1], y * y);
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float temp_alpha = fabsf(alpha[0]) / fmaxf(alpha[1], 1.e-6f);
			float du = temp_alpha * dU_du_crd[i];
			dU_du_crd[i] = 0.f;
			du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			u_crd[i] += du;
		}
	}
}


//������صı����ɲο������kernel����
//ref_crd�Ƕ�Ӧu_crd��node�Ĳο����꣬����vina_atom�ڵ��������������
//atom_to_node_serial
__global__ void Optimize_Structure_BB2_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VECTOR_INT* vina_atom, VECTOR* frc, float* energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
)
{
	//Ϊ���ǿ��ܵļ��٣�������С�ĸ�����Ϣ���ŵ�shared��
	//
	__shared__ float shared_data[23];
	float* rot_matrix = &shared_data[2];
	float* alpha1 = &shared_data[17];
	float* alpha2 = &shared_data[19];
	float* alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//��ʱ������
		shared_data[1] = BIG_ENERGY;//��ʱ������
		//shared_data[2]...shared_data[10]//����ת������
		//shared_data[11] cos_b ����Ӧŷ��ת����
		//shared_data[12] sin_b
		//shared_data[13] cos_a
		//shared_data[14] sin_a
		//shared_data[15] cacb
		//shared_data[16] cbsa
		//shared_data[17] = 0.f;//BB�Ż��õ�alpha
		//shared_data[18] = 0.f;
		//shared_data[19] = 0.f;//BB�Ż��õ�alpha
		//shared_data[20] = 0.f;
		//shared_data[21] = 0.f;//BB�Ż��õ�alpha
		//shared_data[22] = 0.f;
	}

	//������ѭ��ǰ�Ļ�����ʼ��
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		if (i < u_freedom - 6 || u_freedom - 3 < i)
		{
			//u_crd[i] = 0.f;
		}
		last_u_crd[i] = u_crd[i];
	}

	//������ѭ��ǰ����ͬ��
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//�ڵ�ǰ���������¸��¸�ת������
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8];//cacb
				shared_data[16] = rot_matrix[5];//cbsa
			}
		}
		__syncthreads();

		//�ɸ�ת�������ԭʼ�������ɵ�ǰ����
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//����������frc�����ٺ���һ��ͬ��������
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//�����൱��Ҫ��node��a0��Ҫ��ref�����䣬��ѡ����ͬ��ԭ��
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//����ת���Ĳο�ԭ�����ǵ�һ��ԭ�ӣ�rootԭ�ӣ�
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//����ƽ����������
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//�ɵ�ǰ�������node��a��n���ڼ������������ʵ��˳��Ҳ����������ԭ��������У�
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = { vina_atom[node[node_id].root_atom_serial].x,vina_atom[node[node_id].root_atom_serial].y,vina_atom[node[node_id].root_atom_serial].z };
			tempn = { vina_atom[node[node_id].branch_atom_serial].x,vina_atom[node[node_id].branch_atom_serial].y,vina_atom[node[node_id].branch_atom_serial].z };
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//����ʵ�ʲ���Ҫͬ��

		//����ԭ������������
		float total_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR_INT atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VECTOR_INT atom_i = vina_atom[i];
			VECTOR force_i = { 0.f,0.f,0.f };
			VECTOR dr;
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//Ҫ���Ƿ���ԭ��
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.x);//��������ں����⣬���ṩһ������ʸ����ָ������ڷ���
				dr.y = fdimf(box_min.y, atom_i.y);
				dr.z = fdimf(box_min.z, atom_i.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.x, box_max.x);
				dr.y = fdimf(atom_i.y, box_max.y);
				dr.z = fdimf(atom_i.z, box_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				//protein interaction
				VECTOR serial;//�ڵ��ײ�ֵ�����еĸ������
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//�Զ���ֵ����Ҫƫ��������
				//float4 ans = { 0.f,0.f,0.f };
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = { atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z };
				rij = norm3df(dr.x, dr.y, dr.z);//ʸ������
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse;//��Ϊ�������ò�ֵ���ĸ������
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//���Դ���ֵ���ɿ��Ǻ����滻Ϊ���������һά��������protein meshһ��
					//ans = { 0.f,0.f,0.f,0.f };
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//�����־���ʸ������һ���ĶԽ�Ч������ã��������ڵ��������õ������ݾ���������������ϸ���ȷ�����������������Ǹ�˹˥���ģ�����ֻ�൱��ƽ����뱻��Զ��һ�㣩
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w);//���inner list�ǲ��ظ�����pair���õģ�����Ҫ��0.5f

					//LJ test
					/*float r_2 = rij_inverse * rij_inverse;
					float r_4 = r_2 * r_2;
					float r_6 = r_4 * r_2;
					frc_abs = -1.f*(-12.f * 9.4429323e+05f * r_6 + 6.f * 8.0132353e+02f) * r_6 * r_2;
					total_energy_in_thread += 1.f * (9.4429323e+05f * r_6 - 8.0132353e+02f) * r_6;*/

					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		__syncthreads();//�����Ӻ���ȫ�����ݶ��Լ�node�Ĳ�������Ϣ��ȫ

		//��ǰ�˳��Ż�
		if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		{
			if (threadIdx.x == 0)
			{
				energy[0] = shared_data[0];
			}
			//break;
		}
		if (threadIdx.x == 0)
		{
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha1[0] = 0.f;
			alpha1[1] = 0.f;
			alpha2[0] = 0.f;
			alpha2[1] = 0.f;
			alpha3[0] = 0.f;
			alpha3[1] = 0.f;
		}

		//���������
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = { vina_atom[0].x ,vina_atom[0].y , vina_atom[0].z };
			VECTOR temp_crd2 = { vina_atom[i].x ,vina_atom[i].y , vina_atom[i].z };
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		//����BB�Ż�����(��ʱδ��������ת����ƽ���Ͷ�������ɶȵĸ����Ż�)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
			if (i < u_freedom - 6)
			{
				atomicAdd(&alpha1[0], y * s);
				atomicAdd(&alpha1[1], y * y);
			}
			else if (i < u_freedom - 3)
			{
				atomicAdd(&alpha2[0], y * s);
				atomicAdd(&alpha2[1], y * y);
			}
			else
			{
				atomicAdd(&alpha3[0], y * s);
				atomicAdd(&alpha3[1], y * y);
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float du;
			if (i < u_freedom - 6)
			{
				float temp_alpha = fabsf(alpha1[0]) / fmaxf(alpha1[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			}
			else if (i < u_freedom - 3)
			{
				float temp_alpha = fabsf(alpha2[0]) / fmaxf(alpha2[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
			}
			else
			{
				float temp_alpha = fabsf(alpha3[0]) / fmaxf(alpha3[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
			}
			dU_du_crd[i] = 0.f;
			u_crd[i] += du;
		}
		__syncthreads();
	}
}


//��pair���ò�ʹ�ò�ֵ����ֱ�ӽ��м���
__global__ void Optimize_Structure_BB2_Direct_Pair_Device
(
	const int atom_numbers, const int* inner_interaction_list, const float cutoff,
	const int* atom_to_node_serial,
	const VECTOR* ref_crd, VINA_ATOM* vina_atom, VECTOR* frc, float* energy,
	const long long int* protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float* u_crd, float* last_u_crd, float* dU_du_crd, float* last_dU_du_crd,
	const int node_numbers, NODE* node
)
{
	//Ϊ���ǿ��ܵļ��٣�������С�ĸ�����Ϣ���ŵ�shared��
	//
	__shared__ float shared_data[24];
	float* rot_matrix = &shared_data[2];
	float* alpha1 = &shared_data[17];
	float* alpha2 = &shared_data[19];
	float* alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//��ʱ������
		shared_data[1] = BIG_ENERGY;//��ʱ������
		shared_data[23] = 0.f;
	}

	//������ѭ��ǰ�Ļ�����ʼ��
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	//������ѭ��ǰ����ͬ��
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//�ڵ�ǰ���������¸��¸�ת������
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8];//cacb
				shared_data[16] = rot_matrix[5];//cbsa
			}
		}
		__syncthreads();

		//�ɸ�ת�������ԭʼ�������ɵ�ǰ����
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//����������frc�����ٺ���һ��ͬ��������
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//�����൱��Ҫ��node��a0��Ҫ��ref�����䣬��ѡ����ͬ��ԭ��
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//����ת���Ĳο�ԭ�����ǵ�һ��ԭ�ӣ�rootԭ�ӣ�
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].crd.x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//����ƽ����������
			vina_atom[i].crd.y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].crd.z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//�ɵ�ǰ�������node��a��n���ڼ������������ʵ��˳��Ҳ����������ԭ��������У�
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = { vina_atom[node[node_id].root_atom_serial].crd.x,vina_atom[node[node_id].root_atom_serial].crd.y,vina_atom[node[node_id].root_atom_serial].crd.z };
			tempn = { vina_atom[node[node_id].branch_atom_serial].crd.x,vina_atom[node[node_id].branch_atom_serial].crd.y,vina_atom[node[node_id].branch_atom_serial].crd.z };
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//����ʵ�ʲ���Ҫͬ��

		//����ԭ������������
		float total_energy_in_thread = 0.f;
		float intra_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VINA_ATOM atom_j;
			VECTOR temp_force;
			// Modified 2023/08/27: remove dd, dd_ never used
			float rij,  frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VINA_ATOM atom_i = vina_atom[i];
			VECTOR force_i = { 0.f,0.f,0.f };
			VECTOR dr;
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL)//Ҫ���Ƿ���ԭ��
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.crd.x);//��������ں����⣬���ṩһ������ʸ����ָ������ڷ���
				dr.y = fdimf(box_min.y, atom_i.crd.y);
				dr.z = fdimf(box_min.z, atom_i.crd.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.crd.x, box_max.x);
				dr.y = fdimf(atom_i.crd.y, box_max.y);
				dr.z = fdimf(atom_i.crd.z, box_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				//protein interaction
				VECTOR serial;//�ڵ��ײ�ֵ�����еĸ������
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//�Զ���ֵ����Ҫƫ��������
				//ans = { 0.f,0.f,0.f,0.f };
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = { atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z };
				rij = norm3df(dr.x, dr.y, dr.z);//ʸ������
				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					total_energy_in_thread += temp_record;
					intra_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					total_energy_in_thread += temp_record;
					intra_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					total_energy_in_thread += temp_record * surface_distance;
					intra_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						total_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						intra_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						total_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						intra_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		atomicAdd(&shared_data[23], intra_energy_in_thread);
		__syncthreads();//�����Ӻ���ȫ�����ݶ��Լ�node�Ĳ�������Ϣ��ȫ

		//��ǰ�˳��Ż������������Ȼ�����ܶ࣬���Ŀǰֻ�̶ܹ������Ż�����������Ӧ�㹻���ã�
		//if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		//{
		//	//opt_i = MAX_OPTIMIZE_STEPS;
		//	if (threadIdx.x == 0)
		//	{
		//		//energy[0] = shared_data[0];
		//	}
		//	//break;
		//}
		if (threadIdx.x == 0)
		{
			energy[0] = shared_data[0];
			energy[1] = shared_data[23];
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			shared_data[23] = 0.f;
			alpha1[0] = 0.f;
			alpha1[1] = 0.f;
			alpha2[0] = 0.f;
			alpha2[1] = 0.f;
			alpha3[0] = 0.f;
			alpha3[1] = 0.f;
		}

		//���������
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = { vina_atom[0].crd.x ,vina_atom[0].crd.y , vina_atom[0].crd.z };
			VECTOR temp_crd2 = { vina_atom[i].crd.x ,vina_atom[i].crd.y , vina_atom[i].crd.z };
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		//����BB�Ż�����(��ʱδ��������ת����ƽ���Ͷ�������ɶȵĸ����Ż�)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
			if (i < u_freedom - 6)
			{
				atomicAdd(&alpha1[0], y * s);
				atomicAdd(&alpha1[1], y * y);
			}
			else if (i < u_freedom - 3)
			{
				atomicAdd(&alpha2[0], y * s);
				atomicAdd(&alpha2[1], y * y);
			}
			else
			{
				atomicAdd(&alpha3[0], y * s);
				atomicAdd(&alpha3[1], y * y);
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float du;
			if (i < u_freedom - 6)
			{
				float temp_alpha = fabsf(alpha1[0]) / fmaxf(alpha1[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			}
			else if (i < u_freedom - 3)
			{
				float temp_alpha = fabsf(alpha2[0]) / fmaxf(alpha2[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
			}
			else
			{
				float temp_alpha = fabsf(alpha3[0]) / fmaxf(alpha3[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
			}
			dU_du_crd[i] = 0.f;
			u_crd[i] += du;
		}
		__syncthreads();
	}
}
