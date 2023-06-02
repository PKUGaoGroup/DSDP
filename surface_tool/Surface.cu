#include "Surface.cuh"


__global__ void Calculate_Origin_Grid_Occupation_Device
(
	const int atom_numbers, const VECTOR* crd, const float* radius,
	const int grid_numbers, const int layer_numbers, const INT_VECTOR grid_dimension, const float grid_length, const float grid_length_inverse,
	const int extending_numbers, int* grid_occupation
)
{
	const int total_extending_numbers_minus_one = extending_numbers * 2;
	const int total_extending_numbers = extending_numbers * 2 + 1;
	const int total_extending_grid_numbers = total_extending_numbers * total_extending_numbers * total_extending_numbers;
	//对每个原子循环将其贡献加入到网格中
	for (int atom_i = blockIdx.x * blockDim.x + threadIdx.x; atom_i < atom_numbers; atom_i = atom_i + gridDim.x * blockDim.x)
	{
		VECTOR atom_crd = crd[atom_i];
		float atom_radius2 = radius[atom_i] * radius[atom_i];
		VECTOR atom_fraction_crd = { atom_crd.x * grid_length_inverse,atom_crd.y * grid_length_inverse,atom_crd.z * grid_length_inverse };
		INT_VECTOR atom_in_grid_serial = { atom_fraction_crd.x,atom_fraction_crd.y ,atom_fraction_crd.z };

		int dx = 0, dy = 0, dz = 0;
		int x, y, z;

		//需要注意保证遍历不出盒子边界
		for (int neighbor_grid_i = 0; neighbor_grid_i < total_extending_grid_numbers; neighbor_grid_i = neighbor_grid_i + 1)
		{
			z = (atom_in_grid_serial.int_z + dz - extending_numbers);
			y = (atom_in_grid_serial.int_y + dy - extending_numbers);
			x = (atom_in_grid_serial.int_x + dx - extending_numbers);

			//do sth
			VECTOR grid_crd = { grid_length * x,grid_length * y,grid_length * z };
			VECTOR dr = { grid_crd.x - atom_crd.x,grid_crd.y - atom_crd.y ,grid_crd.z - atom_crd.z };
			float dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
			if (dr2 < atom_radius2)
			{
				int neighbor_grid_serial = z * layer_numbers + y * grid_dimension.int_x + x;
				grid_occupation[neighbor_grid_serial] = 1;
			}
			dx = dx + 1;
			dy = dy + (((total_extending_numbers_minus_one - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - total_extending_numbers) >> 31);
			dz = dz + (((total_extending_numbers_minus_one - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - total_extending_numbers) >> 31);
		}
	}
}
__global__ void Calculate_Atom_Near_Surface_Device
(
	const int atom_numbers, const VECTOR* crd, const float* radius, int* is_near_surface,
	const int grid_numbers, const int layer_numbers, const INT_VECTOR grid_dimension, const float grid_length, const float grid_length_inverse,
	const int extending_numbers, const int* surface
)
{
	const int total_extending_numbers_minus_one = extending_numbers * 2;
	const int total_extending_numbers = extending_numbers * 2 + 1;
	const int total_extending_grid_numbers = total_extending_numbers * total_extending_numbers * total_extending_numbers;
	//对每个原子循环将其贡献加入到网格中
	for (int atom_i = blockIdx.x * blockDim.x + threadIdx.x; atom_i < atom_numbers; atom_i = atom_i + gridDim.x * blockDim.x)
	{
		VECTOR atom_crd = crd[atom_i];
		float atom_radius2 = radius[atom_i] * radius[atom_i];
		VECTOR atom_fraction_crd = { atom_crd.x * grid_length_inverse,atom_crd.y * grid_length_inverse,atom_crd.z * grid_length_inverse };
		INT_VECTOR atom_in_grid_serial = { atom_fraction_crd.x,atom_fraction_crd.y ,atom_fraction_crd.z };

		int dx = 0, dy = 0, dz = 0;
		int x, y, z;

		int temp_near_surface = 0;
		//需要注意保证遍历不出盒子边界
		for (int neighbor_grid_i = 0; neighbor_grid_i < total_extending_grid_numbers; neighbor_grid_i = neighbor_grid_i + 1)
		{
			z = (atom_in_grid_serial.int_z + dz - extending_numbers);
			y = (atom_in_grid_serial.int_y + dy - extending_numbers);
			x = (atom_in_grid_serial.int_x + dx - extending_numbers);

			//do sth
			VECTOR grid_crd = { grid_length * x,grid_length * y,grid_length * z };
			VECTOR dr = { grid_crd.x - atom_crd.x,grid_crd.y - atom_crd.y ,grid_crd.z - atom_crd.z };
			float dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
			if (dr2 < atom_radius2)
			{
				int neighbor_grid_serial = z * layer_numbers + y * grid_dimension.int_x + x;
				if (surface[neighbor_grid_serial] == 1)
				{
					temp_near_surface = 1;
					break;
				}
			}
			dx = dx + 1;
			dy = dy + (((total_extending_numbers_minus_one - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - total_extending_numbers) >> 31);
			dz = dz + (((total_extending_numbers_minus_one - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - total_extending_numbers) >> 31);
		}
		is_near_surface[atom_i] = temp_near_surface;
	}
}

__global__ void Smooth_Grid_Occupation_Device
(
	const int grid_numbers, const int layer_numbers, const INT_VECTOR grid_dimension, const INT_VECTOR grid_dimension_minus_one,
	const int* origin_grid_occupation, int* smoothed_grid_occupation
)
{
	INT_VECTOR grid_3d_serial;
	for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < grid_numbers; grid_i = grid_i + gridDim.x * blockDim.x)
	{
		grid_3d_serial.int_x = grid_i % grid_dimension.int_x;
		grid_3d_serial.int_y = (grid_i % layer_numbers) / grid_dimension.int_x;
		grid_3d_serial.int_z = grid_i / layer_numbers;
		int dx = 0, dy = 0, dz = 0;
		int x_, y_, z_, x, y, z, ddx, ddy, ddz, temp_x, temp_y, temp_z;
		int temp_occupation = 0;
		for (int neighbor_grid_i = 0; neighbor_grid_i < 13; neighbor_grid_i = neighbor_grid_i + 1)//只保留对半映射
		{
			ddx = dx - 1;
			ddy = dy - 1;
			ddz = dz - 1;

			z_ = (grid_3d_serial.int_z + ddz);
			y_ = (grid_3d_serial.int_y + ddy);
			x_ = (grid_3d_serial.int_x + ddx);//13个格子中的一个

			z = (grid_3d_serial.int_z - ddz);
			y = (grid_3d_serial.int_y - ddy);
			x = (grid_3d_serial.int_x - ddx);//关于中心格子对称的另一个

			x = x - (x & ((x) >> 31));//如果x<0则x=0，如果x>=0则x=x
			temp_x = (grid_dimension_minus_one.int_x - x);
			x = x + (temp_x & (temp_x >> 31));//如果x<grid_dimension.int_x则x=x，如果x>=grid_dimension.int_x则x=grid_dimension_minus_one.int_x

			y = y - (y & ((y) >> 31));
			temp_y = (grid_dimension_minus_one.int_y - y);
			y = y + (temp_y & (temp_y >> 31));

			z = z - (z & ((z) >> 31));
			temp_z = (grid_dimension_minus_one.int_z - z);
			z = z + (temp_z & (temp_z >> 31));

			x_ = x_ - (x_ & ((x_) >> 31));
			temp_x = (grid_dimension_minus_one.int_x - x_);
			x_ = x_ + (temp_x & (temp_x >> 31));

			y_ = y_ - (y_ & ((y_) >> 31));
			temp_y = (grid_dimension_minus_one.int_y - y_);
			y_ = y_ + (temp_y & (temp_y >> 31));

			z_ = z_ - (z_ & ((z_) >> 31));
			temp_z = (grid_dimension_minus_one.int_z - z_);
			z_ = z_ + (temp_z & (temp_z >> 31));

			//可优化max min
			int grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
			int grid_k = z_ * layer_numbers + y_ * grid_dimension.int_x + x_;
			temp_occupation = (temp_occupation | (origin_grid_occupation[grid_j] & origin_grid_occupation[grid_k]));

			dx = dx + 1;
			dy = dy + (((2 - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - 3) >> 31);
			dz = dz + (((2 - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - 3) >> 31);
		}
		smoothed_grid_occupation[grid_i] = temp_occupation | origin_grid_occupation[grid_i];
	}
}
__global__ void Build_Surface_Device
(
	const int grid_numbers, const int layer_numbers, const INT_VECTOR grid_dimension, const INT_VECTOR grid_dimension_minus_one,
	const int* origin_grid_occupation, int* surface
)
{
	INT_VECTOR grid_3d_serial;
	for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < grid_numbers; grid_i = grid_i + gridDim.x * blockDim.x)
	{
		grid_3d_serial.int_x = grid_i % grid_dimension.int_x;
		grid_3d_serial.int_y = (grid_i % layer_numbers) / grid_dimension.int_x;
		grid_3d_serial.int_z = grid_i / layer_numbers;

		int temp_occupation = 1;
		//邻近6个格子均为1，则自己为0，否则为原始origin_grid_occupation
		int grid_j, x, y, z, temp_z, temp_y, temp_x;
		x = grid_3d_serial.int_x;
		y = grid_3d_serial.int_y;
		z = grid_3d_serial.int_z - 1;
		z = z - (z & ((z) >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.int_x;
		y = grid_3d_serial.int_y - 1;
		z = grid_3d_serial.int_z;
		y = y - (y & ((y) >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.int_x - 1;
		y = grid_3d_serial.int_y;
		z = grid_3d_serial.int_z;
		x = x - (x & ((x) >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		//+1
		x = grid_3d_serial.int_x + 1;
		y = grid_3d_serial.int_y;
		z = grid_3d_serial.int_z;
		temp_x = (grid_dimension_minus_one.int_x - x);
		x = x + (temp_x & (temp_x >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.int_x;
		y = grid_3d_serial.int_y + 1;
		z = grid_3d_serial.int_z;
		temp_y = (grid_dimension_minus_one.int_y - y);
		y = y + (temp_y & (temp_y >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.int_x;
		y = grid_3d_serial.int_y;
		z = grid_3d_serial.int_z + 1;
		temp_z = (grid_dimension_minus_one.int_z - z);
		z = z + (temp_z & (temp_z >> 31));

		grid_j = z * layer_numbers + y * grid_dimension.int_x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		//如果被包围则temp_occupation=1，不然则为0
		surface[grid_i] = origin_grid_occupation[grid_i] & (1 - temp_occupation);//只有当origin_grid_occupation[grid_i]=1且temp_occupation=0才为1
	}
}


void SURFACE::Initial(const float grid_length, const int atom_numbers, const VECTOR* atom_crd, const int* atomic_number)
{
	this->grid_length = grid_length;
	grid_length_inverse = 1.f / grid_length;
	this->atom_crd.clear();
	this->atomic_number.clear();

	VECTOR inner_min = { 100000.f,100000.f,100000.f, }, inner_max = { -100000.f,-100000.f,-100000.f, };
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		inner_min.x = fminf(inner_min.x, atom_crd[i].x);
		inner_min.y = fminf(inner_min.y, atom_crd[i].y);
		inner_min.z = fminf(inner_min.z, atom_crd[i].z);

		inner_max.x = fmaxf(inner_max.x, atom_crd[i].x);
		inner_max.y = fmaxf(inner_max.y, atom_crd[i].y);
		inner_max.z = fmaxf(inner_max.z, atom_crd[i].z);
	}
	move_vec = { -inner_min.x + skin ,-inner_min.y + skin ,-inner_min.z + skin };
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		this->atom_crd.push_back({ atom_crd[i].x + move_vec.x ,atom_crd[i].y + move_vec.y , atom_crd[i].z + move_vec.z });
		this->atomic_number.push_back(atomic_number[i]);
	}
	VECTOR protein_size;
	protein_size.x = inner_max.x - inner_min.x + 2.f * skin;
	protein_size.y = inner_max.y - inner_min.y + 2.f * skin;
	protein_size.z = inner_max.z - inner_min.z + 2.f * skin;

	extending_numbers = ceilf(skin * grid_length_inverse) - 1;

	grid_dimension.int_x = ceilf(protein_size.x / grid_length);
	grid_dimension.int_y = ceilf(protein_size.y / grid_length);
	grid_dimension.int_z = ceilf(protein_size.z / grid_length);
	grid_dimension_minus_one.int_x = grid_dimension.int_x - 1;
	grid_dimension_minus_one.int_y = grid_dimension.int_y - 1;
	grid_dimension_minus_one.int_z = grid_dimension.int_z - 1;


	layer_numbers = grid_dimension.int_x * grid_dimension.int_y;
	grid_numbers = layer_numbers * grid_dimension.int_z;

	box_length.x = grid_length * grid_dimension.int_x;
	box_length.y = grid_length * grid_dimension.int_y;
	box_length.z = grid_length * grid_dimension.int_z;

	//atom radius，来自蒋思远的程序
	atom_radius.clear();
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		if (atomic_number[i] == 1)
		{
			atom_radius.push_back(0.f);
		}
		else if (atomic_number[i] == 6)
		{
			atom_radius.push_back(1.7f);
		}
		else if (atomic_number[i] == 7)
		{
			atom_radius.push_back(1.6f);
		}
		else if (atomic_number[i] == 8)
		{
			atom_radius.push_back(1.55f);
		}
		else if (atomic_number[i] == 16)
		{
			atom_radius.push_back(1.8f);
		}
		else if (atomic_number[i] == 15)
		{
			atom_radius.push_back(1.95f);
		}
		else if (atomic_number[i] == 17)
		{
			atom_radius.push_back(1.8f);
		}
		else if (atomic_number[i] == 9)
		{
			atom_radius.push_back(1.5f);
		}
		else if (atomic_number[i] == 35)
		{
			atom_radius.push_back(1.9f);
		}
		else if (atomic_number[i] == 53)
		{
			atom_radius.push_back(2.1f);
		}
		else if (atomic_number[i] == 34)
		{
			atom_radius.push_back(1.9f);
		}
		else if (atomic_number[i] == 35)
		{
			atom_radius.push_back(2.05f);
		}
		else if (atomic_number[i] == 5)
		{
			atom_radius.push_back(1.8f);
		}
		else if (atomic_number[i] == 20)
		{
			atom_radius.push_back(2.4f);
		}
		else if (atomic_number[i] == 12)
		{
			atom_radius.push_back(2.2f);
		}
		else if (atomic_number[i] == 30)
		{
			atom_radius.push_back(2.1f);
		}
		else if (atomic_number[i] == 100)
		{
			atom_radius.push_back(2.05f);
		}
		atom_radius[i] *= 1.f;//整体调节半径用
	}

	//malloc
	cudaMalloc((void**)&d_atom_crd, sizeof(VECTOR) * atom_numbers);
	cudaMalloc((void**)&d_atom_radius, sizeof(float) * atom_numbers);
	cudaMalloc((void**)&d_atom_is_near_surface, sizeof(int) * atom_numbers);
	cudaMalloc((void**)&d_origin_grid_occupation, sizeof(int) * grid_numbers);
	cudaMalloc((void**)&d_smoothed_grid_occupation, sizeof(int) * grid_numbers);

	//memcpy
	cudaMemcpy(d_atom_crd, &this->atom_crd[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_radius, &this->atom_radius[0], sizeof(float) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemset(d_origin_grid_occupation, 0, sizeof(int) * grid_numbers);
	cudaMemset(d_smoothed_grid_occupation, 0, sizeof(int) * grid_numbers);

	//run
	Calculate_Origin_Grid_Occupation_Device << <64, 64 >> >
		(atom_numbers, d_atom_crd, d_atom_radius,
			grid_numbers, layer_numbers, grid_dimension, grid_length, grid_length_inverse,
			extending_numbers, d_origin_grid_occupation);

	Smooth_Grid_Occupation_Device << <64, 64 >> >
		(
			grid_numbers, layer_numbers, grid_dimension, grid_dimension_minus_one,
			d_origin_grid_occupation, d_smoothed_grid_occupation
			);
	Smooth_Grid_Occupation_Device << <64, 64 >> >
		(
			grid_numbers, layer_numbers, grid_dimension, grid_dimension_minus_one,
			d_smoothed_grid_occupation, d_origin_grid_occupation
			);//做两次，做完后，d_origin_grid_occupation实际为两次smoothed的结果
	Build_Surface_Device << <64, 64 >> >
		(
			grid_numbers, layer_numbers, grid_dimension, grid_dimension_minus_one,
			d_origin_grid_occupation, d_smoothed_grid_occupation
			);//此时d_smoothed_grid_occupation记录的则是表面的网格
	Calculate_Atom_Near_Surface_Device << <64, 64 >> >
		(atom_numbers, d_atom_crd, d_atom_radius, d_atom_is_near_surface,
			grid_numbers, layer_numbers, grid_dimension, grid_length, grid_length_inverse,
			extending_numbers, d_smoothed_grid_occupation);

	atom_is_near_surface.resize(atom_numbers);
	cudaMemcpy(&atom_is_near_surface[0], d_atom_is_near_surface, sizeof(int) * atom_numbers, cudaMemcpyDeviceToHost);
}
