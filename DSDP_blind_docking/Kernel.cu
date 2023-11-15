#include "Kernel.cuh"
__device__ __host__ static void Matrix_Multiply_Vector(VECTOR* __restrict__ c, const float* __restrict__ a, const VECTOR* __restrict__ b)
{
	c[0].x = a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z;
	c[0].y = a[3] * b[0].x + a[4] * b[0].y + a[5] * b[0].z;
	c[0].z = a[6] * b[0].x + a[7] * b[0].y + a[8] * b[0].z;
}

//inner_interaction_list是这样一种结构，用于记录每个原子需要考虑计算同一分子内两体作用的列表
//为方便分配，实际上inner_interaction_list是个atom_numbers*atom_numbers的矩阵
//但每个inner_interaction_list[i*atom_numbers]代表i号原子要考虑两体作用的原子数（存储的考虑编号总是大于i）
//且为了保证效率，要求每一行inner_interaction_list[i*atom_numbers]后面的原子序号都是排了序的。
//frc和energy都会在该kernel里清零重加，因此无需保证输入的frc和energy初始化
//为保持一致性，原子crd坐标均采用VECTOR_INT，int记录的是原子种类。
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
		if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//要求是非氢原子
		{
			//box interaction
			dr.x = fdimf(box_min.x, atom_i.x);//如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
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
			VECTOR serial;//在蛋白插值网格中的格点坐标
			serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
			serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
			serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
			float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//自动插值，需要偏离半个格子
			total_energy_in_thread += ans.w;
			force_i.x += ans.x;
			force_i.y += ans.y;
			force_i.z += ans.z;
		}
		frc[i] = force_i;//该kernel不会在输入的frc上累加
	}
	__syncthreads();//同步，以保证后面两两作用加力时已经全部走过这一步，同时保证能量也成功清零

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
			rij = norm3df(dr.x, dr.y, dr.z);//矢量长度
			if (rij < cutoff)
			{
				rij_inverse = 1.f / (rij + 10.e-6f);
				rij *= pair_potential_grid_length_inverse;//变为两体作用插值表的格点坐标
				dd = rij - floor(rij);
				dd_ = 1.f - dd;
				ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
				frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
				total_energy_in_thread += 0.5f * (dd_ * ans.y + dd * ans.w);//加两次除以2

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

//算力相关的变量可参考上面的kernel函数
//ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
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
	//为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[19];
	float* rot_matrix = &shared_data[2];
	float* alpha = &shared_data[17];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//临时能量项
		shared_data[1] = BIG_ENERGY;//临时能量项
		//shared_data[2]...shared_data[10]//整体转动矩阵
		//shared_data[11] cos_b 均对应欧拉转动角
		//shared_data[12] sin_b
		//shared_data[13] cos_a
		//shared_data[14] sin_a
		//shared_data[15] cacb
		//shared_data[16] cbsa
		//shared_data[17] = 0.f;//BB优化用的alpha
		//shared_data[18] = 0.f;

	}

	//进入主循环前的基本初始化
	for (int i = 0; i < u_freedom; i = i + 1)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	//进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//在当前广义坐标下更新各转动矩阵
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

		//由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//整体平移在最后加上
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
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
		//__syncthreads();//这里实际不需要同步

		//计算原子力和总能量
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
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//要求是非氢原子
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.x);//如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
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
				VECTOR serial;//在蛋白插值网格中的格点坐标
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//自动插值，需要偏离半个格子
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
				rij = norm3df(dr.x, dr.y, dr.z);//矢量长度
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse;//变为两体作用插值表的格点坐标
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w);//如果inner list是不重复计算pair作用的，则不需要乘0.5f

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
		__syncthreads();//能量加和完全，且梯度以及node的叉乘相关信息完全

		//提前退出优化
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

		//计算广义力
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

		//进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
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


//算力相关的变量可参考上面的kernel函数
//ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
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
	//为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float* rot_matrix = &shared_data[2];
	float* alpha1 = &shared_data[17];
	float* alpha2 = &shared_data[19];
	float* alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//临时能量项
		shared_data[1] = BIG_ENERGY;//临时能量项
		//shared_data[2]...shared_data[10]//整体转动矩阵
		//shared_data[11] cos_b 均对应欧拉转动角
		//shared_data[12] sin_b
		//shared_data[13] cos_a
		//shared_data[14] sin_a
		//shared_data[15] cacb
		//shared_data[16] cbsa
		//shared_data[17] = 0.f;//BB优化用的alpha
		//shared_data[18] = 0.f;
		//shared_data[19] = 0.f;//BB优化用的alpha
		//shared_data[20] = 0.f;
		//shared_data[21] = 0.f;//BB优化用的alpha
		//shared_data[22] = 0.f;
	}

	//进入主循环前的基本初始化
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

	//进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//在当前广义坐标下更新各转动矩阵
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

		//由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//整体平移在最后加上
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
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
		//__syncthreads();//这里实际不需要同步

		//计算原子力和总能量
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
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL)//要求是非氢原子
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.x);//如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
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
				VECTOR serial;//在蛋白插值网格中的格点坐标
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//自动插值，需要偏离半个格子
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
				rij = norm3df(dr.x, dr.y, dr.z);//矢量长度
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse;//变为两体作用插值表的格点坐标
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type);//不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
					//ans = { 0.f,0.f,0.f,0.f };
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;//好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w);//如果inner list是不重复计算pair作用的，则不需要乘0.5f

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
		__syncthreads();//能量加和完全，且梯度以及node的叉乘相关信息完全

		//提前退出优化
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

		//计算广义力
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

		//进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
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


//对pair作用不使用插值表，直接进行计算
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
	//为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[24];
	float* rot_matrix = &shared_data[2];
	float* alpha1 = &shared_data[17];
	float* alpha2 = &shared_data[19];
	float* alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;//临时能量项
		shared_data[1] = BIG_ENERGY;//临时能量项
		shared_data[23] = 0.f;
	}

	//进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	//进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		//在当前广义坐标下更新各转动矩阵
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

		//由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x;//这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].crd.x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//整体平移在最后加上
			vina_atom[i].crd.y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].crd.z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		//由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
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
		//__syncthreads();//这里实际不需要同步

		//计算原子力和总能量
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
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL)//要求是非氢原子
			{
				//box interaction
				dr.x = fdimf(box_min.x, atom_i.crd.x);//如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
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
				VECTOR serial;//在蛋白插值网格中的格点坐标
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f);//自动插值，需要偏离半个格子
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
				rij = norm3df(dr.x, dr.y, dr.z);//矢量长度
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
		__syncthreads();//能量加和完全，且梯度以及node的叉乘相关信息完全

		//提前退出优化（开起这个竟然变慢很多，因此目前只能固定次数优化，但理论上应足够够用）
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

		//计算广义力
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

		//进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
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
