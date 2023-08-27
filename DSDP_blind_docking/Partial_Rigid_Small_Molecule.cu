#include "Partial_Rigid_Small_Molecule.cuh"

// Modified 2023/08/27: 
// 1. read HETATM type in ligand pdbqt file
// 2. read more than 99 atoms in pdbqt file
// by CW Dong
void PARTIAL_RIGID_SMALL_MOLECULE::Initial_From_PDBQT(const char* file_name)
{
	//�Ƚ������
	atom_numbers = 0;
	frc.clear();
	//crd.clear();//����Ҫ���
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
	//num_tor//����Ҫ���


	//��pdbqt�ж����������Ϣ����pdbqtͬ����������˵�pdbqt_tree�ĺ���Ҳ������
	NODE temp_node;//���ڲ���node��,����ֻ��ʼ����λ��
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
		}//if ��֤��ǰ��������branch��ʱ���ٶ���һ�ж�����ת��branch��ز���
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
					pdbqt_tree.atom_to_node_serial.push_back(-1);//root�е�ԭ�Ӳ�ָ���κ�node
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
			root_atom_seiral -= 1;//pdbqt��1����ԭ��
			branch_atom_serial -= 1;
			int heavy_atom_numbers = 0;//��¼�ýڵ����ж��ٸ�����ԭ�ӣ������ж��Ƿ����ǻ��༯��
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
					}//�����ǻ�����vina��ʵ�ʲ��������ɶȵĽڵ㣬���Ȼ�������ӽڵ㣬��˱�Ȼ��ENDBRANCH����Ϊpdbqt��tree��������������ȵ�
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

			//����ÿ��branch���������Ը���һ������node��Ϣ
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
			temp_node.a0.z -= crd_from_pdbqt[0].z;//���ں�����õ�origin crd���õ�һ��ԭ�Ӵ�������ԭ��ģ��������Ҫ��ȥ��һ��ԭ������
			temp_node.a = temp_node.a0;
			temp_node.n = temp_node.n0;
			temp_node.last_node_serial = pdbqt_tree.atom_to_node_serial[root_atom_seiral];
			pdbqt_tree.node.push_back(temp_node);

			pdbqt_tree.torsion_numbers += 1;
		}//else if branch
	}
	fclose(in);

	//��pdbqt tree�򻯵�vina tree
	for (int i = 0; i < is_pure_H_freedom[0]; i = i + 1)
	{
		vina_tree.atom_to_node_serial.push_back(-1);
	}//pdbqt�е�rootҲһ����vina�е�root
	for (int i = 0; i < pdbqt_tree.torsion_numbers; i = i + 1)
	{
		int node_serial;
		if (is_pure_H_freedom[i + 1] < 0)
		{
			node_serial = vina_tree.atom_to_node_serial[pdbqt_tree.node[i].root_atom_serial];
		}//if �Ǵ�����ţ���û����ڵ�ԭ�ӹ�����һ�ڵ�Ľڵ���ţ����ڽڵ����¹�������˲���ֱ��ֻ�ýڵ���Ŷ���Ҫ��root ԭ�������ڵ���Ϣ��vina tree�еģ���
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

	//���괦��
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

	//MC��vina�������
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

	//gpu��ԭ�������Ϣ���ڴ����
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

			//MC��vina�������
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

		//MC��vina�������
		cudaMallocHost((void**)&h_vina_atom, sizeof(VINA_ATOM) * atom_numbers);
		cudaMalloc((void**)&d_vina_atom, sizeof(VINA_ATOM) * atom_numbers);

		malloced_atom_numbers = atom_numbers;
	}

	//gpu�����ɶ������Ϣ���ڴ����
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
		cudaMallocHost((void**)&h_u_crd, sizeof(float) * (u_freedom + 1));
		cudaMalloc((void**)&u_crd, sizeof(float) * (u_freedom + 1));
		cudaMallocHost((void**)&h_last_accepted_u_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&dU_du_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&last_u_crd, sizeof(float) * u_freedom);
		cudaMalloc((void**)&last_dU_du_crd, sizeof(float) * u_freedom);
		malloced_u_freedom = u_freedom;
	}

	//gpu�Ͻڵ���Ϣ���ڴ����
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

	//��������ֻ�ǳ�ʼ������������õ�u_crd�������ط����и��£����ȫ��Ϊ0.f
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

	//MC��vina�������
	vina_gpu.last_accepted_energy = input[0].vina_gpu.last_accepted_energy;
	cudaMemcpy(vina_gpu.inner_neighbor_list, input[0].vina_gpu.h_inner_neighbor_list, sizeof(int) * atom_numbers * atom_numbers, cudaMemcpyHostToDevice);
	cudaMemcpy(vina_gpu.d_vina_atom, input[0].vina_gpu.h_vina_atom, sizeof(VINA_ATOM) * atom_numbers, cudaMemcpyHostToDevice);
}

