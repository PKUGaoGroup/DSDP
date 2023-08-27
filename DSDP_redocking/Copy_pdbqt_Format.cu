#include "Copy_pdbqt_Format.cuh"
void COPY_pdbqt_FORMAT::Initial(char* pdbqt_name)
{
	if (pdbqt_content_numbers == -1)
	{
		pdbqt_content.resize(512);
	}

	FILE* pdbqt_file = fopen_safely(pdbqt_name, "r");

	char temp_read_str[256];
	atom_numbers_in_pdbqt = 0;
	pdbqt_content_numbers = 0;
	crd_in_pdbqt.clear();

	while (true)
	{
		char* end_test = fgets(temp_read_str, 256, pdbqt_file);
		if (end_test == NULL)
		{
			break;
		}
		
		pdbqt_content[pdbqt_content_numbers].assign(temp_read_str);
		pdbqt_content_numbers += 1;
		
		if ((temp_read_str[0] == 'A'
			&& temp_read_str[1] == 'T'
			&& temp_read_str[2] == 'O'
			&& temp_read_str[3] == 'M') ||
			(temp_read_str[0] == 'H'
			&& temp_read_str[1] == 'E'
			&& temp_read_str[2] == 'T'
			&& temp_read_str[3] == 'A'
			&& temp_read_str[4] == 'T'
			&& temp_read_str[5] == 'M'))
		{
			VECTOR temp_crd;
			char temp_float_str[9];
			temp_float_str[8] = '\0';
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = temp_read_str[30 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.x);
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = temp_read_str[38 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.y);
			for (int i = 0; i < 8; i = i + 1)
			{
				temp_float_str[i] = temp_read_str[46 + i];
			}
			sscanf(temp_float_str, "%f", &temp_crd.z);
			crd_in_pdbqt.push_back(temp_crd);
			atom_numbers_in_pdbqt += 1;
			pdbqt_atom_list.push_back(1);
		}
		else{pdbqt_atom_list.push_back(0);}
		
		//debug
		//printf("%d ",pdbqt_atom_list[pdbqt_content_numbers-1]);
		//printf("%s",pdbqt_content[pdbqt_content_numbers-1].c_str());
	}
	
	fclose(pdbqt_file);

}

void COPY_pdbqt_FORMAT::Append_Frame_To_Opened_pdbqt(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec)
{
int record2 =0;
	for (int line_number=0;line_number<pdbqt_content_numbers;line_number++)
	{
		if (pdbqt_atom_list[line_number] ==0)
		{ fprintf(pdbqt_file, "%s", pdbqt_content[line_number].c_str());}
		else
		{
		const char* temp_str = pdbqt_content[line_number].c_str();
		fprintf(pdbqt_file, "%30.30s", temp_str);
		VECTOR temp_crd = { 0.f,0.f,0.f };
		temp_crd = crd_in_docking[record2];
		record2 += 1;
		fprintf(pdbqt_file, "%8.3f%8.3f%8.3f", temp_crd.x + move_vec.x, temp_crd.y + move_vec.y, temp_crd.z + move_vec.z);
		fprintf(pdbqt_file, "%s", temp_str + 54);
		}
	}
}
void COPY_pdbqt_FORMAT::Append_Frame_To_Opened_pdbqt_standard(FILE* pdbqt_file, VECTOR* crd_in_docking, const VECTOR move_vec,const int pose_rank,const float score)
{

fprintf(pdbqt_file,"MODEL %d\n",pose_rank);
fprintf(pdbqt_file,"REMARK DSDP RESULT:      %f\n",score);
int record2 =0;
	for (int line_number=0;line_number<pdbqt_content_numbers;line_number++)
	{
		if (pdbqt_atom_list[line_number] ==0)
		{ fprintf(pdbqt_file, "%s", pdbqt_content[line_number].c_str());}
		else
		{
		const char* temp_str = pdbqt_content[line_number].c_str();
		fprintf(pdbqt_file, "%30.30s", temp_str);
		VECTOR temp_crd = { 0.f,0.f,0.f };
		temp_crd = crd_in_docking[record2];
		record2 += 1;
		fprintf(pdbqt_file, "%8.3f%8.3f%8.3f", temp_crd.x + move_vec.x, temp_crd.y + move_vec.y, temp_crd.z + move_vec.z);
		fprintf(pdbqt_file, "%s", temp_str + 54);
		}
	}
	fprintf(pdbqt_file,"ENDMDL\n");
}

		
