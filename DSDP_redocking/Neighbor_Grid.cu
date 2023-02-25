#include "Neighbor_Grid.cuh"
void NEIGHBOR_GRID::Initial(VECTOR box_length, float cutoff, float skin)
{
	this->box_length = box_length;
	this->cutoff = cutoff + skin;
	this->skin = skin;
	skin2 = skin * skin;
	cutoff2 = (this->cutoff) * (this->cutoff);

	grid_dimension.int_x = floorf(box_length.x / cutoff);
	grid_dimension.int_y = floorf(box_length.y / cutoff);
	grid_dimension.int_z = floorf(box_length.z / cutoff);

	grid_numbers = grid_dimension.int_x * grid_dimension.int_y * grid_dimension.int_z;
	layer_numbers = grid_dimension.int_x * grid_dimension.int_y;

	grid_length = { box_length.x / grid_dimension.int_x,box_length.x / grid_dimension.int_x ,box_length.x / grid_dimension.int_x };
	grid_length_inverse = { 1.f / grid_length.x,1.f / grid_length.y ,1.f / grid_length.z };

	neighbor_grid_bucket = (NEIGHBOR_GRID_BUCKET*)malloc(sizeof(NEIGHBOR_GRID_BUCKET) * grid_numbers);

	Clear_Grid_Bucket_Total();

	gpu.neighbor_grid = this;
	cudaMalloc((void**)&gpu.neighbor_grid_bucket, sizeof(NEIGHBOR_GRID_BUCKET) * grid_numbers);
}

void NEIGHBOR_GRID::Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd)
{
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		VECTOR tempcrd = { grid_length_inverse.x * crd[i].x,
		grid_length_inverse.y * crd[i].y,
		grid_length_inverse.z * crd[i].z };
		INT_VECTOR temp_serial = { tempcrd.x,tempcrd.y,tempcrd.z };

		if (temp_serial.int_x < 0 || temp_serial.int_x >= grid_dimension.int_x
			|| temp_serial.int_y < 0 || temp_serial.int_y >= grid_dimension.int_y
			|| temp_serial.int_z < 0 || temp_serial.int_z >= grid_dimension.int_z)
		{
			printf("atom coordinate %f %f %f is out of range of neighbor box\n",
				crd[i].x, crd[i].y, crd[i].z);
			getchar();
		}
		int grid_serial = temp_serial.int_z * layer_numbers + temp_serial.int_y * grid_dimension.int_x + temp_serial.int_x;//000
		neighbor_grid_bucket[grid_serial].atom_serial[neighbor_grid_bucket[grid_serial].atom_serial[0]]
			= i;
		neighbor_grid_bucket[grid_serial].atom_serial[0] += 1;

		if (neighbor_grid_bucket[grid_serial].atom_serial[0] == ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX)
		{
			printf("neighbor_grid_bucket %d max\n", ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX);
			getchar();
		}
	}
}

void NEIGHBOR_GRID::Clear_Grid_Bucket_Total()
{
	for (int i = 0; i < grid_numbers; i = i + 1)
	{
		neighbor_grid_bucket[i].atom_serial[0] = 1;
	}
}


void NEIGHBOR_GRID::GPU::Put_Atom_Into_Grid_Bucket(int atom_numbers, VECTOR* crd)
{
	neighbor_grid->Clear_Grid_Bucket_Total();
	neighbor_grid->Put_Atom_Into_Grid_Bucket(atom_numbers, crd);

	cudaMemcpy(neighbor_grid_bucket, neighbor_grid->neighbor_grid_bucket, sizeof(NEIGHBOR_GRID_BUCKET) * neighbor_grid->grid_numbers, cudaMemcpyHostToDevice);
}