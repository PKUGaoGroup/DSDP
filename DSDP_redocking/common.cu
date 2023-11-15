#include "common.cuh"
bool cmp(INT_FLOAT& a, INT_FLOAT& b)
{
	if (a.energy < b.energy)
	{
		return true;
	}
	else
	{
		return false;
	}
}
FILE* fopen_safely(const char* file_name, const char* mode)
{
	FILE* temp_f = NULL;
	temp_f = fopen(file_name, mode);
	if (temp_f != NULL)
	{
		return temp_f;
	}
	else
	{
		printf("Can't open or create %s!\n", file_name);
		//getchar();
		return NULL;
	}
}
float real_distance(VECTOR& crd, VECTOR& crd2)
{
	float temp_rmsd = 0.f;
	temp_rmsd += (crd.x - crd2.x) * (crd.x - crd2.x);
	temp_rmsd += (crd.y - crd2.y) * (crd.y - crd2.y);
	temp_rmsd += (crd.z - crd2.z) * (crd.z - crd2.z);
	return sqrtf(temp_rmsd);
}
VECTOR unifom_rand_Euler_angles()
{
	//	
	// Cos[��]*Cos[��]	Cos[��]*Sin[��]	-Sin[��]
	//Cos[��] * Sin[��] * Sin[��] - Cos[��] * Sin[��]	Cos[��] * Cos[��] + Sin[��] * Sin[��] * Sin[��]	Cos[��] * Sin[��]
	//	Cos[��] * Cos[��] * Sin[��] + Sin[��] * Sin[��] - (Cos[��] * Sin[��]) + Cos[��] * Sin[��] * Sin[��]	Cos[��] * Cos[��]
	// 
	// 
	// nx^2*(1 - Cos[��]) + Cos[��]	nx*ny*(1 - Cos[��]) - nz*Sin[��]	nx*nz*(1 - Cos[��]) + ny*Sin[��]
	//nx* ny* (1 - Cos[��]) + nz * Sin[��]	ny ^ 2 * (1 - Cos[��]) + Cos[��]	ny * nz * (1 - Cos[��]) - nx * Sin[��]
	//	nx * nz * (1 - Cos[��]) - ny * Sin[��]	ny * nz * (1 - Cos[��]) + nx * Sin[��]	nz ^ 2 * (1 - Cos[��]) + Cos[��]
	// 
	// -Sin[��]=nx*nz*(1 - Cos[��]) + ny*Sin[��]
	// Cos[��]*Sin[��]=nx*ny*(1 - Cos[��]) - nz*Sin[��]
	// Cos[��] * Sin[��]=ny * nz * (1 - Cos[��]) - nx * Sin[��]

	//��sin(theta/2)^2�ܶ�����theta
	float theta = 3.141592654f * rand() / RAND_MAX;
	while (true)
	{
		float f = sinf(0.5f * theta);
		if (f * f > (float)rand() / RAND_MAX)
		{
			break;
		}
		theta = 3.141592654f * rand() / RAND_MAX;
	}
	// �����������ʸ��?
	VECTOR n = {2.f * rand() / RAND_MAX - 1.f, 2.f * rand() / RAND_MAX - 1.f, 2.f * rand() / RAND_MAX - 1.f};
	float dr_1;
	while (true)
	{
		float dr2 = n.x * n.x + n.y * n.y + n.z * n.z;
		if (dr2 > 0.0000001f && dr2 < 1.f)
		{
			dr_1 = 1.f / sqrtf(dr2);
			break;
		}
		n = {2.f * rand() / RAND_MAX - 1.f, 2.f * rand() / RAND_MAX - 1.f, 2.f * rand() / RAND_MAX - 1.f};
	}
	n.x *= dr_1;
	n.y *= dr_1;
	n.z *= dr_1;

	float sin_a = sinf(theta);
	float cos_a = cosf(theta);
	float cos_a_1 = 1.f - cos_a;
	float gamma = atan2f(n.x * n.y * cos_a_1 - n.z * sin_a, n.x * n.x * cos_a_1 + cos_a);
	float beta = atan2f(-n.x * n.z * cos_a_1 - n.y * sin_a, (n.x * n.x * cos_a_1 + cos_a) / cosf(gamma));
	float alpha = atan2f(n.y * n.z * cos_a_1 - n.x * sin_a, n.z * n.z * cos_a_1 + cos_a);

	return {alpha, beta, gamma};
}
float calcualte_heavy_atom_rmsd(const int atom_numbers,const VECTOR* a,const VECTOR* b,const int *atomic_number)
{
	float rmsd = 0.f;
	int heavy_atom_numbers = 0;
	for (int k = 0; k < atom_numbers; k = k + 1)
	{
		if (atomic_number[k] != 1)
		{
			VECTOR dr = { a[k].x - b[k].x, a[k].y - b[k].y, a[k].z - b[k].z };
			rmsd += dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
			heavy_atom_numbers += 1;
		}
	}
	return sqrtf(rmsd / heavy_atom_numbers);
}
int Get_Atomic_Number_From_PDBQT_Atom_Name(const char* atom_name)
{
	if (strcmp(atom_name, "A ") == 0 ||strcmp(atom_name, "C ") == 0)
	{
		return 6;
	}
	else if (strcmp(atom_name, "N ") == 0|| strcmp(atom_name, "NA") == 0)
	{
		return 7;
	}
	else if (strcmp(atom_name, "O ") == 0|| strcmp(atom_name, "OA") == 0)
	{
		return 8;
	}
	else if (strcmp(atom_name, "S ") == 0 ||strcmp(atom_name, "SA") == 0)
	{
		return 16;
	}
	else if (strcmp(atom_name, "P ") == 0)
	{
		return 15;
	}
	else if (strcmp(atom_name, "F ") == 0)
	{
		return 9;
	}
	else if (strcmp(atom_name, "Cl") == 0)
	{
		return 17;
	}
	else if (strcmp(atom_name, "Br") == 0)
	{
		return 35;
	}
	else if (strcmp(atom_name, "I ") == 0)
	{
		return 53;
	}
	else if (strcmp(atom_name, "H ") == 0|| strcmp(atom_name, "HD") == 0)
	{
		return 1;
	}
	else
	{
		printf("find new element %s, please add to program\n",atom_name);
		//getchar();
		return 1;
	}
}

//data is from vina from AD4_parameters
static float Vina_Covalent_Radius(int atomic_number)
{
	if (atomic_number == 6)//C A
	{
		return 0.77f;
	}
	else if (atomic_number == 7)//N NA
	{
		return 0.75f;
	}
	else if (atomic_number == 8)//O OA
	{
		return 0.73f;
	}
	else if (atomic_number == 15)//P
	{
		return 1.06f;
	}
	else if (atomic_number == 16)//S SA
	{
		return 1.02f;
	}
	else if (atomic_number == 1)//H HD
	{
		return 0.37f;
	}
	else if (atomic_number == 9)//F
	{
		return 0.71f;
	}
	else if (atomic_number == 53)//I
	{
		return 1.33f;
	}
	else if (atomic_number == 17)//Cl
	{
		return 0.99f;
	}
	else if (atomic_number == 35)//Br
	{
		return 1.14f;
	}
	else
	{
		printf("In Vina_Covalent_Radius atomic numbers %d is not added\n", atomic_number);
		// Modified 2023/08/27: exit when problem encountered;
		exit(-1);
		//getchar();
	}
}

void Build_Inner_Neighbor_List
(const int atom_numbers, int* neighbor_list, std::vector<VECTOR>& initial_crd, std::vector<int>& atomic_number,
	std::vector<int>& atom_node_serial)
{
	memset(neighbor_list, 0, sizeof(int) * atom_numbers * atom_numbers);
	//����һ��������Ϣ
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		neighbor_list[i * atom_numbers + i] = 100;//�Լ����Լ�Ҳ����Ҫ��
		for (int j = i + 1; j < atom_numbers; j = j + 1)
		{
			//1.1f��Vina���оݱ���һ��
			if (real_distance(initial_crd[i], initial_crd[j]) < 1.1f * (Vina_Covalent_Radius(atomic_number[i]) + Vina_Covalent_Radius(atomic_number[j])))
			{
				neighbor_list[i * atom_numbers + j] = 1;
				neighbor_list[j * atom_numbers + i] = 1;
			}
		}
	}
	//��һ����Ϣ�õ�����������Ϣ���������Ǹ�����˷��������ڵͽ�ʱ��ϡ���ԣ��Ӽ����жϿ��Ա��ⲻ�ټ�������?
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		for (int j = 0; j < atom_numbers; j = j + 1)
		{
			if (neighbor_list[i * atom_numbers + j] == 1)
			{
				for (int k = 0; k < atom_numbers; k = k + 1)
				{
					if (neighbor_list[j * atom_numbers + k] == 1)
					{
						if (neighbor_list[i * atom_numbers + k] == 0)
						{
							neighbor_list[i * atom_numbers + k] = 2;
							neighbor_list[k * atom_numbers + i] = 2;
						}
					}
				}
			}
		}
	}
	//�ɶ��׵õ�����
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		for (int j = 0; j < atom_numbers; j = j + 1)
		{
			if (neighbor_list[i * atom_numbers + j] == 2)
			{
				for (int k = 0; k < atom_numbers; k = k + 1)
				{
					if (neighbor_list[j * atom_numbers + k] == 1)//���ϼ�һ��
					{
						if (neighbor_list[i * atom_numbers + k] == 0)
						{
							neighbor_list[i * atom_numbers + k] = 3;
							neighbor_list[k * atom_numbers + i] = 3;
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		if (atomic_number[i] == 1)
		{
			for (int j = 0; j < atom_numbers; j = j + 1)
			{
				if (neighbor_list[i * atom_numbers + j] == 0)
				{
					neighbor_list[i * atom_numbers + j] = 4;
					neighbor_list[j * atom_numbers + i] = 4;
				}
			}
		}
	}//������ټ���H���ų�

	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		for (int j = i + 1; j < atom_numbers; j = j + 1)
		{
			if (neighbor_list[i * atom_numbers + j] == 0)
			{
				//����ͬһ�������ڲ���Ҳ����Ҫ�㣬��Ͳ��㶼�ǳ���?
				if (atom_node_serial[i] == atom_node_serial[j])
				{
					neighbor_list[i * atom_numbers + j] = 4;
					neighbor_list[j * atom_numbers + i] = 4;
				}
			}
		}
	}

	//��neighbor_list��ʽ����
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		int neighbor_numbers = 0;
		for (int j = i + 1; j < atom_numbers; j = j + 1)
		{
			if (neighbor_list[i * atom_numbers + j] == 0)
			{
				neighbor_list[i * atom_numbers + neighbor_numbers + 1] = j;//i * atom_numbers + neighbor_numbers + 1��ȻС�ڵ���j,��˲�����ڸ���
				neighbor_numbers += 1;
			}
		}
		neighbor_list[i * atom_numbers] = neighbor_numbers;
		//printf("neighbor %d %d %d\n",i,neighbor_numbers,neighbor_list[i * atom_numbers]);
	}
}

static void Refresh_Atom_Type_For_Vina_Atom(const int atom_numbers,VINA_ATOM* vina_atom)
{
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		vina_atom[i].atom_type = -1;
		if (vina_atom[i].radius == 1.9f)
		{
			if (vina_atom[i].is_hydrophobic == 1)
			{
				vina_atom[i].atom_type = 0;
			}
			else
			{
				vina_atom[i].atom_type = 1;
			}
		}
		else if (vina_atom[i].radius == 1.8f && vina_atom[i].is_hydrophobic == 0)
		{
			if (vina_atom[i].is_acceptor == 0 && vina_atom[i].is_donor == 0)
			{
				vina_atom[i].atom_type = 2;
			}
			else if (vina_atom[i].is_acceptor == 0 && vina_atom[i].is_donor == 1)
			{
				vina_atom[i].atom_type = 3;
			}
			else if (vina_atom[i].is_acceptor == 1 && vina_atom[i].is_donor == 0)
			{
				vina_atom[i].atom_type = 4;
			}
			else
			{
				vina_atom[i].atom_type = 5;
			}
		}
		else if (vina_atom[i].radius == 1.7f)
		{
			if (vina_atom[i].is_acceptor == 0 && vina_atom[i].is_donor == 0)
			{
				vina_atom[i].atom_type = 6;
			}
			else if (vina_atom[i].is_acceptor == 0 && vina_atom[i].is_donor == 1)
			{
				vina_atom[i].atom_type = 7;
			}
			else if (vina_atom[i].is_acceptor == 1 && vina_atom[i].is_donor == 0)
			{
				vina_atom[i].atom_type = 8;
			}
			else
			{
				vina_atom[i].atom_type = 9;
			}
		}
		else if (vina_atom[i].radius == 2.f && vina_atom[i].is_hydrophobic == 0)
		{
			vina_atom[i].atom_type = 10;
		}
		else if (vina_atom[i].radius == 2.1f)
		{
			vina_atom[i].atom_type = 11;
		}
		else if (vina_atom[i].radius == 1.5f)
		{
			vina_atom[i].atom_type = 12;
		}
		else if (vina_atom[i].radius == 1.8f)
		{
			vina_atom[i].atom_type = 13;
		}
		else if (vina_atom[i].radius == 2.0f)
		{
			vina_atom[i].atom_type = 14;
		}
		else if (vina_atom[i].radius == 2.2f)
		{
			vina_atom[i].atom_type = 15;
		}
		else if (vina_atom[i].radius == 1.2f)
		{
			vina_atom[i].atom_type = 16;
		}
		else if (vina_atom[i].radius == -100.f)
		{
			vina_atom[i].atom_type = 17;
		}
		if (vina_atom[i].atom_type == -1)
		{
			printf("waring wrong vina atom meeted\n");
			//getchar();
		}
	}
}
int Build_Vina_Atom(VINA_ATOM* vina_atom, std::vector<int>& atom_type, std::vector<VECTOR>& initial_crd, std::vector<int>& atomic_number)
{

	int atom_numbers = atom_type.size();

	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		vina_atom[i].is_donor = 0;//��������ѭ����i����ʱʵ�ʻ��j�������ã�Ҫ�ȳ�ʼ����
	}
	char* atom_type_char = NULL;
	VINA_ATOM temp_vina_atom;
	//Ŀǰ����ɸѡ�߼��£����ھ��󲿷���ϵ���жϽ����vina����ȫһ�µģ��ص�������ˮC�����O,N���ж�
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		temp_vina_atom.crd = initial_crd[i];
		//temp_vina_atom.charge = 0.f;//��ʱû��ʹ��
		temp_vina_atom.is_donor = vina_atom[i].is_donor;
		temp_vina_atom.is_acceptor = 0;
		temp_vina_atom.is_hydrophobic = 0;
		temp_vina_atom.radius = 0.f;

		atom_type_char = (char*)&atom_type[i];
		if (strcmp(atom_type_char, "A ") == 0 ||
			strcmp(atom_type_char, "C ") == 0)
		{
			temp_vina_atom.radius = 1.9f;
			//����Ƿ�����ԭ������?
			int is_za = 0;
			for (int j = 0; j < atom_numbers; j = j + 1)
			{
				if (atomic_number[j] != 1 && atomic_number[j] != 6)//self is 6
				{
					if (real_distance(temp_vina_atom.crd, initial_crd[j]) < 1.1f * (0.77f + Vina_Covalent_Radius(atomic_number[j])))
					{
						is_za = 1;
						break;
					}
				}
			}
			if (is_za == 0)
			{
				temp_vina_atom.is_hydrophobic = 1;
			}
		}
		else if (strcmp(atom_type_char, "N ") == 0 ||
			strcmp(atom_type_char, "NA") == 0)
		{
			temp_vina_atom.radius = 1.8f;
			if (strcmp(atom_type_char, "NA") == 0)
			{
				temp_vina_atom.is_acceptor = 1;
			}
		}
		else if (
			strcmp(atom_type_char, "OA") == 0)
		{
			temp_vina_atom.radius = 1.7f;
			temp_vina_atom.is_acceptor = 1;
		}
		else if (
			strcmp(atom_type_char, "P ") == 0)
		{
			temp_vina_atom.radius = 2.1f;
		}
		else if (
			strcmp(atom_type_char, "S ") == 0 ||
			strcmp(atom_type_char, "SA") == 0)//vina��û����ΪSA�������������?
		{
			temp_vina_atom.radius = 2.0f;
		}
		else if (
			strcmp(atom_type_char, "F ") == 0)
		{
			temp_vina_atom.radius = 1.5f;
			temp_vina_atom.is_hydrophobic = 1;
		}
		else if (
			strcmp(atom_type_char, "Cl") == 0)
		{
			temp_vina_atom.radius = 1.8f;
			temp_vina_atom.is_hydrophobic = 1;
		}
		else if (
			strcmp(atom_type_char, "Br") == 0)
		{
			temp_vina_atom.radius = 2.0f;
			temp_vina_atom.is_hydrophobic = 1;
		}
		else if (
			strcmp(atom_type_char, "I ") == 0)
		{
			temp_vina_atom.radius = 2.2f;
			temp_vina_atom.is_hydrophobic = 1;
		}
		else if (
			strcmp(atom_type_char, "H ") == 0 ||
			strcmp(atom_type_char, "HD") == 0)
		{
			temp_vina_atom.radius = -100.f;
			if (strcmp(atom_type_char, "HD") == 0)
			{
				for (int j = 0; j < atom_numbers; j = j + 1)
				{
					if (atomic_number[j] == 7 || atomic_number[j] == 8)//ֻ��N O������H donor
					{
						if (real_distance(temp_vina_atom.crd, initial_crd[j]) < 1.1f * (0.37f + Vina_Covalent_Radius(atomic_number[j])))
						{
							vina_atom[j].is_donor = 1;
						}
					}
				}
			}//��donor
		}
		else
		{
			printf("new ad4 atom type %s found in pdbqt\n", atom_type_char);
			//getchar();
			return 1;
		}
		vina_atom[i] = temp_vina_atom;
	}//for atom i

	Refresh_Atom_Type_For_Vina_Atom(atom_numbers, vina_atom);
	return 0;
}

float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b)
{
	float energy = 0.f, frc_abs = 0.f;
	float dr = real_distance(a.crd, b.crd);
	if (dr < 8.f && a.atom_type < 17 && b.atom_type < 17)
	{
		float surface_distance = dr - a.radius - b.radius;
		float temp_record;

		//gauss1
		temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
		energy += temp_record;
		frc_abs += 2.f * k_gauss1_2 * temp_record * surface_distance;

		//gauss2
		float dp = surface_distance - k_gauss2_c;
		temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
		energy += temp_record;
		frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

		//repulsion
		temp_record = k_repulsion * surface_distance * signbit(surface_distance);
		energy += temp_record * surface_distance;
		frc_abs += -2.f * temp_record;

		//hydrophobic
		if ((a.is_hydrophobic & b.is_hydrophobic))
		{
			temp_record = 1.f * k_hydrophobic;
			energy += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
			frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
		}

		//h_bond
		if (((a.is_donor & b.is_acceptor) | (a.is_acceptor & b.is_donor)))
		{
			temp_record = 1.f * k_h_bond;
			energy += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
			frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
		}
	}
	return { frc_abs,energy };
}
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b,const float dr)
{
	float energy = 0.f, frc_abs = 0.f;
	if (dr < 8.f && a.atom_type < 17 && b.atom_type < 17)
	{
		float surface_distance = dr - a.radius - b.radius;
		float temp_record;

		//gauss1
		temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
		energy += temp_record;
		frc_abs += 2.f * k_gauss1_2 * temp_record * surface_distance;

		//gauss2
		float dp = surface_distance - k_gauss2_c;
		temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
		energy += temp_record;
		frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

		//repulsion
		temp_record = k_repulsion * surface_distance * signbit(surface_distance);
		energy += temp_record * surface_distance;
		frc_abs += -2.f * temp_record;

		//hydrophobic
		if ((a.is_hydrophobic & b.is_hydrophobic))
		{
			temp_record = 1.f * k_hydrophobic;
			energy += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
			frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
		}

		//h_bond
		if (((a.is_donor & b.is_acceptor) | (a.is_acceptor & b.is_donor)))
		{
			temp_record = 1.f * k_h_bond;
			energy += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
			frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
		}
	}
	return { frc_abs,energy };
}

void Read_Atom_Line_In_PDBQT(const char* line, std::vector<VECTOR>& crd, std::vector<float>& charge, std::vector<int>& atom_type)
{
	char temp_str[9];

	//��ȡpdbqt�е�ԭ��������Ϣ�����һ�У�?
	temp_str[0] = line[77];
	temp_str[1] = line[78];
	temp_str[2] = '\0';
	int* temp_atom_type = (int*)temp_str;
	atom_type.push_back(temp_atom_type[0]);

	//��ȡpdbqtԭ������
	VECTOR temp_crd;
	temp_str[8] = '\0';
	memcpy(temp_str, &line[30], sizeof(char) * 8);
	sscanf(temp_str, "%f", &temp_crd.x);
	memcpy(temp_str, &line[38], sizeof(char) * 8);
	sscanf(temp_str, "%f", &temp_crd.y);
	memcpy(temp_str, &line[46], sizeof(char) * 8);
	sscanf(temp_str, "%f", &temp_crd.z);
	crd.push_back(temp_crd);

	//��ȡ���?
	float temp_charge;
	sscanf(&line[67], "%f", &temp_charge);
	charge.push_back(18.2223f * temp_charge);//amber��λ���ɴ˿ɱ�֤����r��λΪ��ʱ��������q*q/rʱ��λΪkcal/mol
}
