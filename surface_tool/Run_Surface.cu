#include "Surface.cuh"
#include "omp.h"
SURFACE sur;
int main(int argn,char *argv[])
{
//double zhout=omp_get_wtime();
	FILE* in = NULL; 
	FILE* out = NULL;
	for (int i = 0; i < argn; i = i + 1)
	{
		if (strcmp(argv[i], "-i") == 0)
		{
			i = i + 1;
			in= fopen(argv[i], "r");
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			i = i + 1;
			out = fopen(argv[i], "w");
		}
	}
	if (in == NULL)
	{
		printf("please input -i %s to select a pdb file\n");
		getchar();
		return 0;
	}
	if (out == NULL)
	{
		printf("no correct -o %s out file name input, use default name SURFACE_OUT.txt\n");
		out = fopen("SURFACE_OUT.txt", "w");
	}

	std::vector<VECTOR>crd;
	std::vector<float>charge;
	std::vector<int>atom_type;
	std::vector<int>atomic_number;
	char line[256];
	char str_segment[256];
	while (true)
	{
		char* end_test = fgets(line, 256, in);
		if (end_test == NULL)
		{
			break;
		}
		else if ((line[0] == 'A' && line[1] == 'T' && line[2] == 'O' && line[3] == 'M' && line[77] != 'H')
			|| (line[0] == 'H' && line[1] == 'E' && line[2] == 'T' && line[3] == 'A'  && line[77] != 'H'))//垃圾原子也要如此读入
		{
			str_segment[0] = line[77];
			str_segment[1] = line[78];
			str_segment[2] = '\0';
			atomic_number.push_back(Get_Atomic_Number_From_PDBQT_Atom_Name(str_segment));
			Read_Atom_Line_In_PDBQT(line, crd, charge, atom_type);
		}
	}
	fclose(in);

	//第一个参数0.1f指定小格子的大小（单位：埃）
	sur.Initial(2.0f, crd.size(), &crd[0], &atomic_number[0]);
	for (int i = 0; i < crd.size(); i = i + 1)
	{
		fprintf(out,"%d\n",sur.atom_is_near_surface[i]);
	}
	fclose(out);
	//printf("%lf",omp_get_wtime()-zhout);
	return 0;
}
