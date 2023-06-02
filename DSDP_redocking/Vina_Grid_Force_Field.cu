#include "Vina_Grid_Force_Field.cuh"

//vina atom type
// X-Score
//const sz XS_TYPE_C_H = 0;//_H指代疏水 ，基本只有当C纯连接C或H时，该C成为C_H
//const sz XS_TYPE_C_P = 1;
//const sz XS_TYPE_N_P = 2;
//const sz XS_TYPE_N_D = 3;
//const sz XS_TYPE_N_A = 4;
//const sz XS_TYPE_N_DA = 5;
//const sz XS_TYPE_O_P = 6;
//const sz XS_TYPE_O_D = 7;
//const sz XS_TYPE_O_A = 8;
//const sz XS_TYPE_O_DA = 9;
//const sz XS_TYPE_S_P = 10;
//const sz XS_TYPE_P_P = 11;
//const sz XS_TYPE_F_H = 12;
//const sz XS_TYPE_Cl_H = 13;
//const sz XS_TYPE_Br_H = 14;
//const sz XS_TYPE_I_H = 15;
//const sz XS_TYPE_Met_D = 16;//这个类型应该是金属原子通用类型
//const sz XS_TYPE_SIZE = 17;//在程序中所有氢原子都是该类型
static std::vector<VINA_ATOM> Return_Vina_Atom_List()
{
	std::vector<VINA_ATOM> vina_atom;
	vina_atom.clear();
	VINA_ATOM temp;

	//0
	temp.atom_type = 0;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 1;
	temp.radius = 1.9f;
	vina_atom.push_back(temp);

	temp.atom_type = 1;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 1.9f;
	vina_atom.push_back(temp);

	temp.atom_type = 2;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 1.8f;
	vina_atom.push_back(temp);

	temp.atom_type = 3;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 1;
	temp.is_hydrophobic = 0;
	temp.radius = 1.8f;
	vina_atom.push_back(temp);

	temp.atom_type = 4;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 1;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 1.8f;
	vina_atom.push_back(temp);

	temp.atom_type = 5;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 1;
	temp.is_donor = 1;
	temp.is_hydrophobic = 0;
	temp.radius = 1.8f;
	vina_atom.push_back(temp);

	temp.atom_type = 6;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 1.7f;
	vina_atom.push_back(temp);

	temp.atom_type = 7;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 1;
	temp.is_hydrophobic = 0;
	temp.radius = 1.7f;
	vina_atom.push_back(temp);

	temp.atom_type = 8;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 1;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 1.7f;
	vina_atom.push_back(temp);

	temp.atom_type = 9;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 1;
	temp.is_donor = 1;
	temp.is_hydrophobic = 0;
	temp.radius = 1.7f;
	vina_atom.push_back(temp);

	temp.atom_type = 10;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 2.f;
	vina_atom.push_back(temp);

	temp.atom_type = 11;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = 2.1f;
	vina_atom.push_back(temp);

	temp.atom_type = 12;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 1;
	temp.radius = 1.5f;
	vina_atom.push_back(temp);

	temp.atom_type = 13;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 1;
	temp.radius = 1.8f;
	vina_atom.push_back(temp);

	temp.atom_type = 14;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 1;
	temp.radius = 2.0f;
	vina_atom.push_back(temp);

	temp.atom_type = 15;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 1;
	temp.radius = 2.2f;
	vina_atom.push_back(temp);

	temp.atom_type = 16;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 1;
	temp.is_hydrophobic = 0;
	temp.radius = 1.2f;
	vina_atom.push_back(temp);

	//该原子种类实际不参与作用，是H的实际归属
	temp.atom_type = 17;
	temp.crd = { 0.f,0.f,0.f };
	temp.is_acceptor = 0;
	temp.is_donor = 0;
	temp.is_hydrophobic = 0;
	temp.radius = -100.f;
	vina_atom.push_back(temp);

	return vina_atom;
}
void VINA_GRID_FORCE_FIELD::Initial(const int grid_numbers_in_one_dimemsion, const float cutoff)
{
	basic_vina_atom = Return_Vina_Atom_List();
	cudaMalloc((void**)&d_basic_vina_atom, sizeof(VINA_ATOM) * type_numbers);
	cudaMemcpy(d_basic_vina_atom,&basic_vina_atom[0],sizeof(VINA_ATOM)*type_numbers,cudaMemcpyHostToDevice);

	this->cutoff = cutoff;
	grid_dimension.int_x = grid_numbers_in_one_dimemsion;
	grid_dimension.int_y = grid_numbers_in_one_dimemsion;
	grid_dimension.int_z = grid_numbers_in_one_dimemsion;
	layer_numbers = grid_dimension.int_x * grid_dimension.int_y;
	grid_numbers = layer_numbers * grid_dimension.int_z;
	grid.Cuda_Texture_Initial(this);
}

void VINA_GRID_FORCE_FIELD::GRID::Cuda_Texture_Initial(VINA_GRID_FORCE_FIELD* vgff)
{
	vina_grid_force_field = vgff;

	cudaMalloc((void**)&potential, sizeof(float4) * vina_grid_force_field->grid_numbers * vina_grid_force_field->type_numbers);
	printf("size of protein mesh %d\n", sizeof(float4) * vina_grid_force_field->grid_numbers * vina_grid_force_field->type_numbers);
	cudaMalloc((void**)&texObj_for_kernel, sizeof(long long int) * vina_grid_force_field->type_numbers);
	cudaArray_potential.resize(vina_grid_force_field->type_numbers);
	copyParams_potential.resize(vina_grid_force_field->type_numbers);
	texObj_potential.resize(vina_grid_force_field->type_numbers);

	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 32, 32, 32,
			cudaChannelFormatKindFloat);
	cudaExtent cuEx;
	cuEx.depth = vina_grid_force_field->grid_dimension.int_z;
	cuEx.height = vina_grid_force_field->grid_dimension.int_y;
	cuEx.width = vina_grid_force_field->grid_dimension.int_x;
	cudaMemcpy3DParms temp_cudaMemcpy3DParms = { 0 };

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;//cudaFilterModePoint cudaFilterModeLinear
	texDesc.readMode = cudaReadModeElementType;
	texDesc.borderColor[0] = 0.f;
	texDesc.borderColor[1] = 0.f;
	texDesc.borderColor[2] = 0.f;
	texDesc.borderColor[3] = 10.e6f;//出边界时给一个高能量，希望这些构象总是被拒接
	texDesc.normalizedCoords = 0;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	for (int i = 0; i < vina_grid_force_field->type_numbers; i = i + 1)
	{
		cudaMalloc3DArray(&cudaArray_potential[i], &channelDesc, cuEx);

		float4* temp_ptr = &potential[i * vina_grid_force_field->grid_numbers];
		temp_cudaMemcpy3DParms.srcPtr = make_cudaPitchedPtr((void*)temp_ptr, cuEx.width * sizeof(float4), cuEx.width, cuEx.height);
		temp_cudaMemcpy3DParms.dstArray = cudaArray_potential[i];
		temp_cudaMemcpy3DParms.extent = cuEx;
		temp_cudaMemcpy3DParms.kind = cudaMemcpyDeviceToDevice;

		copyParams_potential[i] = temp_cudaMemcpy3DParms;

		resDesc.res.array.array = cudaArray_potential[i];
		cudaCreateTextureObject(&texObj_potential[i], &resDesc, &texDesc, NULL);
	}
	cudaMemcpy(texObj_for_kernel, &texObj_potential[0], sizeof(long long int) * vina_grid_force_field->type_numbers, cudaMemcpyHostToDevice);
}
void VINA_GRID_FORCE_FIELD::GRID::Copy_Potential_To_Texture()
{
	for (int i = 0; i < vina_grid_force_field->type_numbers; i = i + 1)
	{
		cudaMemcpy3D(&copyParams_potential[i]);
	}
}

//这个kernel计算时对应调用整张显卡资源，非stream类型kernel
static __global__ void Calculate_Protein_Potential_Grid_Device
(
	const int grid_numbers, const INT_VECTOR grid_dimension, const int type_numbers, float4* potential,
	const VECTOR grid_min, const VECTOR grid_length,
	const float cutoff2,const VINA_ATOM * basic_vina_atom,
	const int atom_numbers, const VINA_ATOM* protein_vina,
	const VECTOR neighbor_grid_length_inverse, const INT_VECTOR neighbor_grid_dimension, const NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket
)
{
	INT_VECTOR grid_3d_serial;
	int layer_numbers = grid_dimension.int_x * grid_dimension.int_y;
	int neighbor_layer_numbers = neighbor_grid_dimension.int_x * neighbor_grid_dimension.int_y;
	VECTOR grid_crd;
	VECTOR temp_crd;
	INT_VECTOR temp_serial;
	int dx, dy, dz;
	int x, y, z;
	for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < grid_numbers; grid_i = grid_i + gridDim.x * blockDim.x)
	{
		grid_3d_serial.int_x = grid_i % grid_dimension.int_x;
		grid_3d_serial.int_y = (grid_i % layer_numbers) / grid_dimension.int_x;
		grid_3d_serial.int_z = grid_i / layer_numbers;

		grid_crd.x = grid_length.x * grid_3d_serial.int_x + grid_min.x;
		grid_crd.y = grid_length.y * grid_3d_serial.int_y + grid_min.y;
		grid_crd.z = grid_length.z * grid_3d_serial.int_z + grid_min.z;

		temp_crd.x = neighbor_grid_length_inverse.x * grid_crd.x;
		temp_crd.y = neighbor_grid_length_inverse.y * grid_crd.y;
		temp_crd.z = neighbor_grid_length_inverse.z * grid_crd.z;

		temp_serial = { (int)temp_crd.x,(int)temp_crd.y ,(int)temp_crd.z };

		dx = 0, dy = 0, dz = 0;
		for (int neighbor_grid_i = 0; neighbor_grid_i < 27; neighbor_grid_i = neighbor_grid_i + 1)
		{
			z = (temp_serial.int_z + dz - 1);
			y = (temp_serial.int_y + dy - 1);
			x = (temp_serial.int_x + dx - 1);
			int neighbor_grid_serial = z * neighbor_layer_numbers + y * neighbor_grid_dimension.int_x + x;
			for (int i = 1; i < neighbor_grid_bucket[neighbor_grid_serial].atom_serial[0]; i = i + 1)
			{
				int atom_i = neighbor_grid_bucket[neighbor_grid_serial].atom_serial[i];
				VINA_ATOM vina_atom = protein_vina[atom_i];
				VECTOR dr = { vina_atom.crd.x - grid_crd.x,vina_atom.crd.y - grid_crd.y, vina_atom.crd.z - grid_crd.z };
				float dis2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

				if (dis2 < cutoff2)
				{
					dis2 = sqrtf(dis2);
					float dr_abs_inverse = 1.f / fmaxf(dis2, 1.e-6f);
					dr.x *= dr_abs_inverse;
					dr.y *= dr_abs_inverse;
					dr.z *= dr_abs_inverse;

					for (int type_i = 0; type_i < type_numbers; type_i = type_i + 1)
					{
						VINA_ATOM basic_atom = basic_vina_atom[type_i];
						float surface_distance = dis2 - vina_atom.radius - basic_atom.radius;
						float temp_record;
						float energy = 0.f;
						float frc_abs = 0.f;
						temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
						energy += temp_record;
						frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

						float dp = surface_distance - k_gauss2_c;
						temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
						energy += temp_record;
						frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

						temp_record = k_repulsion * surface_distance * signbit(surface_distance);
						energy += temp_record * surface_distance;
						frc_abs += -2.f * temp_record;

						if ((vina_atom.is_hydrophobic & basic_atom.is_hydrophobic))
						{
							temp_record = 1.f * k_hydrophobic;
							energy += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
						}

						if (((vina_atom.is_donor & basic_atom.is_acceptor) | (vina_atom.is_acceptor & basic_atom.is_donor)))
						{
							temp_record = 1.f * k_h_bond;
							energy += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
						}

						atomicAdd(&potential[(size_t)type_i * grid_numbers + grid_i].w, energy);
						atomicAdd(&potential[(size_t)type_i * grid_numbers + grid_i].x, -frc_abs * dr.x);//符号是因为前面的dr矢量是蛋白原子坐标减去格点坐标
						atomicAdd(&potential[(size_t)type_i * grid_numbers + grid_i].y, -frc_abs * dr.y);
						atomicAdd(&potential[(size_t)type_i * grid_numbers + grid_i].z, -frc_abs * dr.z);
					}
				}
			}
			dx = dx + 1;
			dy = dy + (((2 - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - 3) >> 31);
			dz = dz + (((2 - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - 3) >> 31);
		}
	}
}


void VINA_GRID_FORCE_FIELD::GRID::Calculate_Protein_Potential_Grid
(
	const VECTOR box_min, const float box_length,
	const int atom_numbers, const VINA_ATOM* d_protein,
	const VECTOR neighbor_grid_length_inverse, const INT_VECTOR neighbor_grid_dimension, const NEIGHBOR_GRID_BUCKET* neighbor_grid_bucket
)
{
	cudaMemset(potential, 0, sizeof(float4) * vina_grid_force_field->grid_numbers * vina_grid_force_field->type_numbers);

	//为保证大部分情况下，分子在随机扰动时，任意原子均不会走出插值网格的区域，需要格子比本身的限制区域大一些
	vina_grid_force_field->grid_min = { box_min.x,box_min.y ,box_min.z };
	vina_grid_force_field->box_length = { box_length ,box_length ,box_length  };
	vina_grid_force_field->grid_length = {
		vina_grid_force_field->box_length.x / (vina_grid_force_field->grid_dimension.int_x - 1) ,
	vina_grid_force_field->box_length.y / (vina_grid_force_field->grid_dimension.int_y - 1) ,
	vina_grid_force_field->box_length.z / (vina_grid_force_field->grid_dimension.int_z - 1) };

	vina_grid_force_field->grid_length_inverse = {
		1.f / vina_grid_force_field->grid_length.x,
		1.f / vina_grid_force_field->grid_length.y,
		1.f / vina_grid_force_field->grid_length.z
	};
	Calculate_Protein_Potential_Grid_Device
		<< <64, 64 >> >
		(
			vina_grid_force_field->grid_numbers, vina_grid_force_field->grid_dimension, vina_grid_force_field->type_numbers, potential,
			vina_grid_force_field->grid_min, vina_grid_force_field->grid_length,
			vina_grid_force_field->cutoff * vina_grid_force_field->cutoff, vina_grid_force_field->d_basic_vina_atom,
			atom_numbers, d_protein,
			neighbor_grid_length_inverse, neighbor_grid_dimension, neighbor_grid_bucket
			);
	Copy_Potential_To_Texture();
}
