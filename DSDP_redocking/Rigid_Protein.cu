#include "Rigid_Protein.cuh"
void RIGID_PROTEIN::Initial_Protein_From_PDBQT(const char* file_name,const VECTOR box_length)
{
	crd.clear();
	charge.clear();
	atomic_number.clear();
	atom_type.clear();

	FILE* in = fopen_safely(file_name, "r");
	char str_line[256];
	char str_segment[256] = { '\0' };
	while (true)
	{
		char* end_test = fgets(str_line, 256, in);
		if (end_test == NULL)
		{
			break;
		}
		sscanf(str_line, "%s", str_segment);
		if (str_segment[0]=='A'
			&& str_segment[1]=='T'
			&& str_segment[2] == 'O'
			&& str_segment[3] == 'M'
			)
		{
			str_segment[0] = str_line[77];
			str_segment[1] = str_line[78];
			str_segment[2] = '\0';
			atomic_number.push_back(Get_Atomic_Number_From_PDBQT_Atom_Name(str_segment));

			VECTOR temp_crd;
			char temp_float_str[9];
			temp_float_str[8] = '\0';
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = str_line[30 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.x);
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = str_line[38 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.y);
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = str_line[46 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.z);
			crd.push_back(temp_crd);

			float temp_charge;
			sscanf(&str_line[67], "%f", &temp_charge);
			charge.push_back(18.2223f * temp_charge);

			int atom_serial;
			sscanf(&str_line[7], "%d", &atom_serial);
			atom_serial = atom_serial - 1;

			char temp_c[4];
			temp_c[0] = str_line[77];
			temp_c[1] = str_line[78];
			temp_c[2] = '\0';
			int* a = (int*)temp_c;
			atom_type.push_back(a[0]);
		}
	}
	fclose(in);
	atom_numbers = crd.size();
	VECTOR need_box_length=Find_A_Proper_Box();//经过此步后，蛋白坐标已经经过平移，由move_vec记录
	if (need_box_length.x > box_length.x
		|| need_box_length.y > box_length.y
		|| need_box_length.z > box_length.z)
	{
		printf("protein with skin size %f %f %f is bigger than simulation box size %f %f %f\n",
			need_box_length.x, need_box_length.y, need_box_length.z,
			box_length.x, box_length.y, box_length.z);
		//getchar();
	}


	//vina_atom
	vina_atom.resize(atom_numbers);
	if (malloced_atom_numbers >= atom_numbers)
	{
		;
	}
	else
	{
		if (d_vina_atom != NULL)
		{
			cudaFree(d_vina_atom);
		}
		cudaMalloc((void**)&d_vina_atom, sizeof(VINA_ATOM) * atom_numbers);
		malloced_atom_numbers = atom_numbers;
	}
	Build_Vina_Atom(&vina_atom[0], atom_type, crd, atomic_number);
	cudaMemcpy(d_vina_atom, &vina_atom[0], sizeof(VINA_ATOM) * atom_numbers, cudaMemcpyHostToDevice);
}

VECTOR RIGID_PROTEIN::Find_A_Proper_Box()
{
	VECTOR inner_min = { 100000.f,100000.f,100000.f, }, inner_max = { -100000.f,-100000.f,-100000.f, };
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		inner_min.x = fminf(inner_min.x, crd[i].x);
		inner_min.y = fminf(inner_min.y, crd[i].y);
		inner_min.z = fminf(inner_min.z, crd[i].z);

		inner_max.x = fmaxf(inner_max.x, crd[i].x);
		inner_max.y = fmaxf(inner_max.y, crd[i].y);
		inner_max.z = fmaxf(inner_max.z, crd[i].z);
	}
	move_vec = { -inner_min.x + skin.x ,-inner_min.y + skin.y ,-inner_min.z + skin.z };
	for (int i = 0; i < atom_numbers; i = i + 1)
	{
		crd[i].x = crd[i].x + move_vec.x;
		crd[i].y = crd[i].y + move_vec.y;
		crd[i].z = crd[i].z + move_vec.z;
	}
	protein_size.x = inner_max.x - inner_min.x + 2.f * skin.x;
	protein_size.y = inner_max.y - inner_min.y + 2.f * skin.y;
	protein_size.z = inner_max.z - inner_min.z + 2.f * skin.z;
	return protein_size;
}
