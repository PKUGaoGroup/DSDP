#include "Partial_Rigid_Small_Molecule.cuh"

// Modified 2023/08/27: 
// 1. read HETATM type in ligand pdbqt file
// 2. read more than 99 atoms in pdbqt file
// by CW Dong
void PARTIAL_RIGID_SMALL_MOLECULE::Initial_From_PDBQT(const char* file_name)
{
	//????????
	atom_numbers = 0;
	frc.clear();
	//crd.clear();//????????
	crd_from_pdbqt.clear();
	//origin_crd
	//move_vec
	atomic_number.clear();
	atom_mass.clear();
	charge.clear();
	atom_type.clear();
	pdbqt_tree.torsion_numbers = 0;
	pdbqt_tree.node.clear();
	pdbqt_tree.atom_to_node_serial.clear();
	is_pure_H_freedom.clear();
	vina_tree.torsion_numbers = 0;
	vina_tree.node.clear();
	vina_tree.atom_to_node_serial.clear();
	//num_tor//????????


	//??pdbqt?��???????????????pdbqt?????????????pdbqt_tree????????????
	NODE temp_node;//???????node??,????????????��??
	memset(temp_node.matrix, 0, sizeof(float) * 9);
	temp_node.matrix[0] = 1.f, temp_node.matrix[4] = 1.f, temp_node.matrix[8] = 1.f;
	FILE* in = fopen_safely(file_name, "r");
	char str_line[256];
	char str_segment[256] = { '\0' };
	while (true)
	{
		if (strcmp(str_segment, "BRANCH") != 0)
		{
			char* end_test = fgets(str_line, 256, in);
			if (end_test == NULL)
			{
				break;
			}
		}//if ??????????????branch????????????��????????branch??????
		sscanf(str_line, "%s", str_segment);

		if (strcmp(str_segment, "ROOT") == 0)
		{
			// Modified 2023/08/27: no warning now
			while (fgets(str_line, 256, in))
			{
				sscanf(str_line, "%s", str_segment);
				if (strcmp(str_segment, "ENDROOT") == 0)
				{
					is_pure_H_freedom.push_back((atom_numbers));
					break;
				}
				// Modified 2023/08/27: 'HETATM'
				else if (strcmp(str_segment, "ATOM") == 0 || strcmp(str_segment, "HETATM") == 0)
				{
					Read_Atom_Line_In_PDBQT(str_line, crd_from_pdbqt, charge, atom_type);
					atomic_number.push_back(Get_Atomic_Number_From_PDBQT_Atom_Name((char*)&atom_type[atom_numbers]));
					//atom_mass.push_back(bd[0].Element_Mass(atomic_number[atom_numbers]));
					pdbqt_tree.atom_to_node_serial.push_back(-1);//root?��?????????��?node
					atom_numbers += 1;
				}
				else
				{
					printf("unexpected line in pdbqt:\n%s\n", str_segment);
					exit(-1);
					//getchar();
				}//if atom
			}//while in root
		}//if root
		else if (strcmp(str_segment, "BRANCH") == 0)
		{
			int root_atom_seiral;
			int branch_atom_serial;
			// Modified 2023/08/27: [6] instead of [8], read more than 99 atoms
			sscanf(&str_line[6], " %d %d", &root_atom_seiral, &branch_atom_serial);
			root_atom_seiral -= 1;//pdbqt??1???????
			branch_atom_serial -= 1;
			int heavy_atom_numbers = 0;//??????????��?????????????????��?????????????
			// Modified 2023/08/27: no warning now
			while (fgets(str_line, 256, in))
			{
				sscanf(str_line, "%s", str_segment);
				if (strcmp(str_segment, "BRANCH") == 0 || strcmp(str_segment, "ENDBRANCH") == 0)
				{
					if (strcmp(str_segment, "ENDBRANCH") == 0
						&& heavy_atom_numbers == 1)
					{
						is_pure_H_freedom.push_back(-(atom_numbers));
					}//???????????vina???????????????????????????????????????ENDBRANCH?????pdbqt??tree????????????????
					else
					{
						is_pure_H_freedom.push_back((atom_numbers));
					}
					break;
				}
				// Modified 2023/08/27: 'HETATM'
				else if (strcmp(str_segment, "ATOM") == 0 || strcmp(str_segment, "HETATM") == 0)
				{
					Read_Atom_Line_In_PDBQT(str_line, crd_from_pdbqt, charge, atom_type);
					atomic_number.push_back(Get_Atomic_Number_From_PDBQT_Atom_Name((char*)&atom_type[atom_numbers]));
					//atom_mass.push_back(bd[0].Element_Mass(atomic_number[atom_numbers]));
					pdbqt_tree.atom_to_node_serial.push_back(pdbqt_tree.torsion_numbers);
					if (atomic_number[atom_numbers] != 1)
					{
						heavy_atom_numbers += 1;
					}
					atom_numbers += 1;
				}
				else
				{
					printf("unexpected line in pdbqt:\n%s\n", str_segment);
					exit(-1);
					//getchar();
				}//if atom
			}//while in a branch

			//???????branch????????????????????node???
			temp_node.root_atom_serial = root_atom_seiral;
			temp_node.branch_atom_serial = branch_atom_serial;
			temp_node.a0 = crd_from_pdbqt[root_atom_seiral];
			temp_node.n0 = crd_from_pdbqt[branch_atom_serial];
			temp_node.n0.x -= temp_node.a0.x;
			temp_node.n0.y -= temp_node.a0.y;
			temp_node.n0.z -= temp_node.a0.z;
			float temp_length = 1.f / sqrtf(temp_node.n0.x * temp_node.n0.x + temp_node.n0.y * temp_node.n0.y + temp_node.n0.z * temp_node.n0.z);
			temp_node.n0.x *= temp_length;
			temp_node.n0.y *= temp_length;
			temp_node.n0.z *= temp_length;
			temp_node.a0.x -= crd_from_pdbqt[0].x;
			temp_node.a0.y -= crd_from_pdbqt[0].y;
			temp_node.a0.z -= crd_from_pdbqt[0].z;//???????????origin crd??????????????????????????????????????????????
			temp_node.a = temp_node.a0;
			temp_node.n = temp_node.n0;
			temp_node.last_node_serial = pdbqt_tree.atom_to_node_serial[root_atom_seiral];
			pdbqt_tree.node.push_back(temp_node);

			pdbqt_tree.torsion_numbers += 1;
		}//else if branch
	}
	fclose(in);

	//??pdbqt tree???vina tree
	for (int i = 0; i < is_pure_H_freedom[0]; i = i + 1)
	{
		vina_tree.atom_to_node_serial.push_back(-1);
	}//pdbqt?��?root??????vina?��?root
	for (int i = 0; i < pdbqt_tree.torsion_numbers; i = i + 1)
	{
		int node_serial;
		if (is_pure_H_freedom[i + 1] < 0)
		{
			node_serial = vina_tree.atom_to_node_serial[pdbqt_tree.node[i].root_atom_serial];
		}//if ???????????????????????????????????????????????????????????????????????????root ???????????????vina tree?��????
		else
		{
			temp_node = pdbqt_tree.node[i];
			temp_node.last_node_serial = vina_tree.atom_to_node_serial[temp_node.root_atom_serial];
			vina_tree.node.push_back(temp_node);

			node_serial = vina_tree.torsion_numbers;
			vina_tree.torsion_numbers += 1;
		}
		for (int j = abs(is_pure_H_freedom[i]); j < abs(is_pure_H_freedom[i + 1]); j = j + 1)
		{
			vina_tree.atom_to_node_serial.push_back(node_serial);
		}
	}

	num_tor = (float)0.5f * (pdbqt_tree.torsion_numbers + vina_tree.torsion_numbers);

	//??????
	origin_crd.resize(atom_numbers);
	move_vec = crd_from_pdbqt[0];
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		origin_crd[i].x = crd_from_pdbqt[i].x - move_vec.x;
		origin_crd[i].y = crd_from_pdbqt[i].y - move_vec.y;
		origin_crd[i].z = crd_from_pdbqt[i].z - move_vec.z;
	}
	crd = origin_crd;
	frc.resize(atom_numbers);
	memset(&frc[0], 0, sizeof(VECTOR) * atom_numbers);

	vina_gpu.Initial(this, &vina_tree);
	pdbqt_gpu.Initial(this, &pdbqt_tree);

	//MC??vina???????
	vina_gpu.last_accepted_energy = 100.f;
	Build_Inner_Neighbor_List
	(atom_numbers, vina_gpu.h_inner_neighbor_list, crd_from_pdbqt, atomic_number,
		vina_tree.atom_to_node_serial);
	cudaMemcpy(vina_gpu.inner_neighbor_list, vina_gpu.h_inner_neighbor_list, sizeof(int)*atom_numbers*atom_numbers, cudaMemcpyHostToDevice);
	Build_Vina_Atom(vina_gpu.h_vina_atom, atom_type, crd_from_pdbqt, atomic_number);
	cudaMemcpy(vina_gpu.d_vina_atom, vina_gpu.h_vina_atom, sizeof(VINA_ATOM)* atom_numbers, cudaMemcpyHostToDevice);
}

void PARTIAL_RIGID_SMALL_MOLECULE::GPU::Initial(PARTIAL_RIGID_SMALL_MOLECULE* mol, const TREE* tree)
{
	partial_rigid_small_molecule = mol;
	atom_numbers = partial_rigid_small_molecule[0].atom_numbers;
	node_numbers = tree[0].torsion_numbers;
	u_freedom = node_numbers + 6;

	//gpu???????????????????
	if (malloced_atom_numbers >= partial_rigid_small_molecule[0].atom_numbers)
	{
		;
	}
	else
	{
		if (origin_crd != NULL)
		{
			cudaFree(origin_crd);
			cudaFree(ref_crd);
			cudaFree(crd);
			cudaFree(last_crd);
			cudaFree(frc);
			cudaFree(atom_to_node_serial);
			cudaFree(inner_neighbor_list);
			free(h_inner_neighbor_list);

			//MC??vina???????
			cudaFreeHost(h_vina_atom);
			cudaFree(d_vina_atom);
		}
		cudaMalloc((void**)&origin_crd, sizeof(VECTOR) * atom_numbers);
		cudaMalloc((void**)&ref_crd, sizeof(VECTOR) * atom_numbers);
		cudaMalloc((void**)&crd, sizeof(VECTOR) * atom_numbers);
		cudaMalloc((void**)&last_crd, sizeof(VECTOR) * atom_numbers);
		cudaMalloc((void**)&frc, sizeof(VECTOR) * atom_numbers);
		cudaMalloc((void**)&atom_to_node_serial, sizeof(int) * atom_numbers);
		cudaMalloc((void**)&inner_neighbor_list, sizeof(int) * atom_numbers * atom_numbers);
		h_inner_neighbor_list = (int*)malloc(sizeof(int) * atom_numbers * atom_numbers);

		//MC??vina???????
		cudaMallocHost((void**)&h_vina_atom, sizeof(VINA_ATOM) * atom_numbers);
		cudaMalloc((void**)&d_vina_atom, sizeof(VINA_ATOM) * atom_numbers);

		malloced_atom_numbers = atom_numbers;
	}

	//gpu?????????????????????
	if (malloced_u_freedom >= u_freedom)
	{
		;
	}
	else
	{
		if (u_crd != NULL)
		{
			cudaFreeHost(h_u_crd);
			cudaFree(u_crd);
			cudaFreeHost(h_last_accepted_u_crd);
			cudaFree(dU_du_crd);
			cudaFree(last_u_crd);
			cudaFree(last_dU_du_crd);
		}
		cudaMallocHost((void**)&h_u_crd, sizeof(float) * (u_freedom + 2));
		cudaMalloc((void**)&u_crd, sizeof(float) * (u_freedom + 2));
		cudaMallocHost((void**)&h_last_accepted_u_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&dU_du_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&last_u_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&last_dU_du_crd, sizeof(float) * u_freedom);
		malloced_u_freedom = u_freedom;
	}

	//gpu???????????????
	if (malloced_node_numbers >= node_numbers)
	{
		;
	}
	else
	{
		if (node != NULL)
		{
			cudaFree(node);
		}
		cudaMalloc((void**)&node, sizeof(NODE) * node_numbers);

		malloced_u_freedom = node_numbers;
	}


	cudaMemcpy(origin_crd, &partial_rigid_small_molecule[0].origin_crd[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(ref_crd, &partial_rigid_small_molecule[0].origin_crd[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(crd, &partial_rigid_small_molecule[0].crd[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(last_crd, &partial_rigid_small_molecule[0].crd[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(frc, &partial_rigid_small_molecule[0].frc[0], sizeof(VECTOR) * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(atom_to_node_serial, &tree[0].atom_to_node_serial[0], sizeof(int) * atom_numbers, cudaMemcpyHostToDevice);

	//???????????????????????????u_crd????????????��???????????0.f
	memset(&h_u_crd[0], 0, sizeof(float) * u_freedom);
	cudaMemset(u_crd, 0, sizeof(float) * u_freedom);
	cudaMemset(last_u_crd, 0, sizeof(float) * u_freedom);
	memset(h_last_accepted_u_crd, 0, sizeof(float) * u_freedom);
	cudaMemset(dU_du_crd, 0, sizeof(float) * u_freedom);
	cudaMemset(last_dU_du_crd, 0, sizeof(float) * u_freedom);

	cudaMemcpy(node, &tree[0].node[0], sizeof(NODE) * node_numbers, cudaMemcpyHostToDevice);

}

void PARTIAL_RIGID_SMALL_MOLECULE::Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(PARTIAL_RIGID_SMALL_MOLECULE* input)
{
	atom_numbers = input[0].atom_numbers;
	frc.resize(atom_numbers);
	memset(&frc[0], 0, sizeof(VECTOR) * atom_numbers);
	crd = input[0].crd;
	crd_from_pdbqt = input[0].crd_from_pdbqt;
	origin_crd = input[0].origin_crd;
	move_vec = input[0].move_vec;
	atomic_number = input[0].atomic_number;
	atom_mass = input[0].atom_mass;
	charge = input[0].charge;
	atom_type = input[0].atom_type;

	pdbqt_tree.atom_to_node_serial = input[0].pdbqt_tree.atom_to_node_serial;
	pdbqt_tree.torsion_numbers = input[0].pdbqt_tree.torsion_numbers;
	pdbqt_tree.node = input[0].pdbqt_tree.node;

	is_pure_H_freedom = input[0].is_pure_H_freedom;

	vina_tree.atom_to_node_serial = input[0].vina_tree.atom_to_node_serial;
	vina_tree.torsion_numbers = input[0].vina_tree.torsion_numbers;
	vina_tree.node = input[0].vina_tree.node;

	num_tor = input[0].num_tor;

	vina_gpu.Initial(this, &vina_tree);
	pdbqt_gpu.Initial(this, &pdbqt_tree);

	//MC??vina???????
	vina_gpu.last_accepted_energy = input[0].vina_gpu.last_accepted_energy;
	cudaMemcpy(vina_gpu.inner_neighbor_list, input[0].vina_gpu.h_inner_neighbor_list, sizeof(int) * atom_numbers * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(vina_gpu.d_vina_atom, input[0].vina_gpu.h_vina_atom, sizeof(VINA_ATOM) * atom_numbers, cudaMemcpyHostToDevice);

	//if need calulate on CPU with openmp
	cudaMemcpy(vina_gpu.h_inner_neighbor_list, input[0].vina_gpu.h_inner_neighbor_list, sizeof(int) * atom_numbers * atom_numbers, cudaMemcpyHostToHost);
	cudaMemcpy(vina_gpu.h_vina_atom, input[0].vina_gpu.h_vina_atom, sizeof(VINA_ATOM) * atom_numbers, cudaMemcpyHostToHost);
}


void PARTIAL_RIGID_SMALL_MOLECULE::Refresh_origin_crd(const VECTOR * new_crd)
{
	memcpy(&origin_crd[0],new_crd,sizeof(VECTOR)*atom_numbers);
	move_vec=new_crd[0];

	//refresh VINA_TREE
	for(int node_i =0 ; node_i < vina_tree.torsion_numbers; node_i += 1 )
	{

		vina_tree.node[node_i].branch_atom_serial;
		vina_tree.node[node_i].root_atom_serial;

		vina_tree.node[node_i].a0 = origin_crd[vina_tree.node[node_i].root_atom_serial];

		vina_tree.node[node_i].n0 = origin_crd[vina_tree.node[node_i].branch_atom_serial];
		vina_tree.node[node_i].n0.x -= vina_tree.node[node_i].a0.x;
		vina_tree.node[node_i].n0.y -= vina_tree.node[node_i].a0.y;
		vina_tree.node[node_i].n0.z -= vina_tree.node[node_i].a0.z;
		float temp_length = 1.f / sqrtf(vina_tree.node[node_i].n0.x * vina_tree.node[node_i].n0.x + vina_tree.node[node_i].n0.y * vina_tree.node[node_i].n0.y + vina_tree.node[node_i].n0.z * vina_tree.node[node_i].n0.z);
		vina_tree.node[node_i].n0.x *=temp_length;
		vina_tree.node[node_i].n0.y *=temp_length;
		vina_tree.node[node_i].n0.z *=temp_length;

		vina_tree.node[node_i].a0.x -= move_vec.x;
		vina_tree.node[node_i].a0.y -= move_vec.y;
		vina_tree.node[node_i].a0.z -= move_vec.z;

		vina_tree.node[node_i].a = vina_tree.node[node_i].a0;
		vina_tree.node[node_i].n = vina_tree.node[node_i].n0;
	}

	for(int i=0;i<atom_numbers;i=i+1)
	{
		origin_crd[i].x-=move_vec.x;
		origin_crd[i].y-=move_vec.y;
		origin_crd[i].z-=move_vec.z;
	}

	memcpy(&crd[0],&origin_crd[0],sizeof(VECTOR)*atom_numbers);

}


void PARTIAL_RIGID_SMALL_MOLECULE::Build_Neighbor_List(const float cutoff_with_skin,const int protein_atom_numbers, const VECTOR *protein_crd, const int * protein_atomic_number)
{
	neighbor_list.resize(atom_numbers);
	if(neighbor_list_total==NULL)
	{
		neighbor_list_total=(int*)malloc(sizeof(int)*atom_numbers*MAX_NEIGHBOR_NUMBERS);
	}
	else if(vina_gpu.malloced_atom_numbers < atom_numbers)
	{
		free(neighbor_list_total);
		neighbor_list_total=(int*)malloc(sizeof(int)*atom_numbers*MAX_NEIGHBOR_NUMBERS);
	}
	for(int i=0;i<atom_numbers;i=i+1)
	{
		neighbor_list[i]=&neighbor_list_total[size_t(i*MAX_NEIGHBOR_NUMBERS)];
	}

	for(int ligand_atom_i=0; ligand_atom_i < atom_numbers; ligand_atom_i +=1)
	{
		neighbor_list[ligand_atom_i][0]=0;
		for(int protein_atom_j=0; protein_atom_j < protein_atom_numbers; protein_atom_j +=1)
		{
			if(protein_atomic_number[protein_atom_j]==1)
			{
				continue;
			}
			VECTOR temp_i=crd[ligand_atom_i];
			temp_i.x += move_vec.x;
			temp_i.y += move_vec.y;
			temp_i.z += move_vec.z;
			VECTOR temp_j=protein_crd[protein_atom_j];


			float distance_i_j = (temp_i.x - temp_j.x)*(temp_i.x - temp_j.x) + (temp_i.y - temp_j.y)*(temp_i.y - temp_j.y) + (temp_i.z - temp_j.z)*(temp_i.z - temp_j.z);

			if(distance_i_j < cutoff_with_skin*cutoff_with_skin)
			{
				neighbor_list[ligand_atom_i][0]+=1;
				neighbor_list[ligand_atom_i][neighbor_list[ligand_atom_i][0]]=protein_atom_j;


			}
		}
	}


}
		__host__ static void Matrix_Multiply_Vector(VECTOR* __restrict__ c, const float* __restrict__ a, const VECTOR* __restrict__ b)
		{
			c[0].x = a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z;
			c[0].y = a[3] * b[0].x + a[4] * b[0].y + a[5] * b[0].z;
			c[0].z = a[6] * b[0].x + a[7] * b[0].y + a[8] * b[0].z;
		}

float PARTIAL_RIGID_SMALL_MOLECULE::Refine_Structure(const VINA_ATOM *protein,const VECTOR*protein_crd)
{
	int u_freedom=vina_gpu.u_freedom;
	int node_numbers=vina_gpu.node_numbers;
		dU_du_crd.resize(u_freedom);
		u_crd.resize(u_freedom);
		memset(&u_crd[0],0,sizeof(float)*u_freedom);
	float rot_matrix[9];
	float shared_data[24];
	float score_out=10000.f;
	for(int opt_step=0;opt_step<100;opt_step+=1)
	{
		//calculate rot matrix
		for (int i = 0; i <= node_numbers; i = i + 1)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = vina_tree.node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				vina_tree.node[i].matrix[0] = temp_matrix_1[0];
				vina_tree.node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				vina_tree.node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				vina_tree.node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				vina_tree.node[i].matrix[4] = temp_matrix_1[4];
				vina_tree.node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				vina_tree.node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				vina_tree.node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				vina_tree.node[i].matrix[8] = temp_matrix_1[8];
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


		for (int i = 0; i < atom_numbers; i = i + 1)
		{
			int current_node_id = vina_tree.atom_to_node_serial[i];
			frc[i] = { 0.f,0.f,0.f };//??????????frc?????????????????????
			VECTOR temp_crd1 = origin_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = origin_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - vina_tree.node[current_node_id].a0.x;//???????????node??a0?????ref???????????????????
				temp_crd2.y = temp_crd1.y - vina_tree.node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - vina_tree.node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, vina_tree.node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += vina_tree.node[current_node_id].a0.x;
				temp_crd1.y += vina_tree.node[current_node_id].a0.y;
				temp_crd1.z += vina_tree.node[current_node_id].a0.z;

				current_node_id = vina_tree.node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x;//???????????????????????????root????
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			crd[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x;//???????????????
			crd[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			crd[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}

		for (int node_id = 0; node_id < node_numbers; node_id = node_id + 1)
			{
				float temp_length;
				VECTOR tempa, tempn;
				tempa = { crd[vina_tree.node[node_id].root_atom_serial].x,crd[vina_tree.node[node_id].root_atom_serial].y,crd[vina_tree.node[node_id].root_atom_serial].z };
				tempn = { crd[vina_tree.node[node_id].branch_atom_serial].x,crd[vina_tree.node[node_id].branch_atom_serial].y,crd[vina_tree.node[node_id].branch_atom_serial].z };
				tempn.x -= tempa.x;
				tempn.y -= tempa.y;
				tempn.z -= tempa.z;
				temp_length = 1.f/sqrtf(tempn.x*tempn.x+ tempn.y*tempn.y + tempn.z*tempn.z);
				tempn.x *= temp_length;
				tempn.y *= temp_length;
				tempn.z *= temp_length;
				vina_tree.node[node_id].n = tempn;
				vina_tree.node[node_id].a = tempa;
			}

		//calculate force and score
		float energy=0.f;
		score_out=0.f;
		for(int atom_i =0; atom_i < atom_numbers; atom_i +=1 )
		{
			VECTOR ligand_crd_i=crd[atom_i];
			ligand_crd_i.x += move_vec.x;
			ligand_crd_i.y += move_vec.y;
			ligand_crd_i.z += move_vec.z;
			frc[atom_i]={0.f,0.f,0.f};
			//start from 1, because the 0-th is total_numbers
			//printf("atom %d\n",atom_i);
			for(int neighbor_j =1; neighbor_j <=neighbor_list[atom_i][0]; neighbor_j +=1)
			{
				
				int protein_atom_j=neighbor_list[atom_i][neighbor_j];
				//printf("neighbor %d protein %d\n",neighbor_j,protein_atom_j);
				VECTOR dr={ligand_crd_i.x - protein_crd[protein_atom_j].x,ligand_crd_i.y - protein_crd[protein_atom_j].y,ligand_crd_i.z - protein_crd[protein_atom_j].z};
				float dr_abs =sqrtf(dr.x*dr.x+dr.y*dr.y+dr.z*dr.z);
				float2 ans= Vina_Pair_Interaction(vina_gpu.h_vina_atom[atom_i], protein[protein_atom_j], dr_abs);
				energy += ans.y;
				score_out += ans.y;
				frc[atom_i].x+=ans.x*dr.x;
				frc[atom_i].y+=ans.x*dr.y;
				frc[atom_i].z+=ans.x*dr.z;
			}
			int inner_list_start=atom_i*atom_numbers;
			int inner_numbers =vina_gpu.h_inner_neighbor_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{	
				int atom_j = vina_gpu.h_inner_neighbor_list[inner_list_start + k];

				VECTOR dr={ crd[atom_i].x - crd[atom_j].x ,crd[atom_i].y - crd[atom_j].y,crd[atom_i].z - crd[atom_j].z};
				float dr_abs =sqrtf(dr.x*dr.x+dr.y*dr.y+dr.z*dr.z);
				float2 ans= Vina_Pair_Interaction(vina_gpu.h_vina_atom[atom_i], vina_gpu.h_vina_atom[atom_j], dr_abs);
				energy+=ans.y;
				frc[atom_i].x+=ans.x*dr.x;
				frc[atom_i].y+=ans.x*dr.y;
				frc[atom_i].z+=ans.x*dr.z;

				frc[atom_j].x-=ans.x*dr.x;
				frc[atom_j].y-=ans.x*dr.y;
				frc[atom_j].z-=ans.x*dr.z;				
			}
		}
		//printf("%f\n",energy);
		
		
		
		//calculate grad u_crd
		memset(&dU_du_crd[0],0,sizeof(float)*u_freedom);
		for (int i = 0; i < atom_numbers; i = i + 1)
		{
			VECTOR center = { crd[0].x ,crd[0].y , crd[0].z };
			VECTOR temp_crd2 = { crd[i].x ,crd[i].y , crd[i].z };
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			dU_du_crd[u_freedom - 1]+=(temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y);
			dU_du_crd[u_freedom - 2]+= (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]);
			dU_du_crd[u_freedom - 3]+= (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12]));

			dU_du_crd[u_freedom - 6]+= temp_frc.x;
			dU_du_crd[u_freedom - 5]+= temp_frc.y;
			dU_du_crd[u_freedom - 4]+= temp_frc.z;

			int current_node_id = vina_tree.atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - vina_tree.node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - vina_tree.node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - vina_tree.node[current_node_id].a.z;
				rot_axis = vina_tree.node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				dU_du_crd[current_node_id]+= (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z);
				current_node_id = vina_tree.node[current_node_id].last_node_serial;
			}
		}

		for(int i=0;i<u_freedom;i=i+1)
		{
			u_crd[i]+=  0.00001f*dU_du_crd[i];
		}
	}
	return score_out;
}