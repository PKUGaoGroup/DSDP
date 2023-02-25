#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include <string>
#include <unordered_map>
#include <vector>
struct FLOAT17
{
	float data[17];
};
char SPACE_to_UNDERSCORE(char c)
{
	if (c == ' ')
	{
		return '_';
	}
	else
	{
		return c;
	}
}
void Replace_COMMA_to_SPACE(char* str)
{
	while (*str != '\0')
	{
		if (*str == ',')
		{
			*str = ' ';
		}
		str++;
	}
}
int main(int argn, char* argv[])
{
	FILE* data_in = NULL;
	FILE* pdb_in = NULL;
	FILE* out = NULL;
	char line[256];
	char str_segment[256];

	for (int i = 0; i < argn; i = i + 1)
	{
		if (strcmp(argv[i], "-i") == 0)
		{
			i = i + 1;
			pdb_in = fopen(argv[i], "r");
		}
		else if(strcmp(argv[i], "-o") == 0)
		{
			i = i + 1;
			out = fopen(argv[i], "w");
		}
		else if (strcmp(argv[i], "-data") == 0)
		{
			i = i + 1;
			data_in = fopen(argv[i], "r");
		}
	}
	if (pdb_in == NULL || out == NULL)
	{
		printf("please input -i %s to select a pdb file and -o %s a output name\n");
		getchar();
		return 1;
	}
	if (data_in == NULL)
	{
		data_in = fopen("17_FEATURES_DATA.txt", "r");
		if (data_in == NULL)
		{
			printf("the default data file 17_FEATURES_DATA.txt is not found\n");
			//getchar();
			return 1;
		}
	}

	//初始化data
	std::unordered_map<std::string, int>data_map;
	std::vector<FLOAT17>data_save;
	fgets(line, 256, data_in);//第一行只是描述
	int atom_type_numbers = 0;
	while (true)
	{
		int pan = fscanf(data_in, "%s", str_segment);
		if (pan == EOF)
		{
			break;
		}
		std::string temp_atom_name(str_segment);
		data_map.insert({ temp_atom_name ,atom_type_numbers });
		FLOAT17 temp_float17;
		for (int i = 0; i < 17; i = i + 1)
		{
			fscanf(data_in, "%f", &temp_float17.data[i]);
		}
		data_save.push_back(temp_float17);
		atom_type_numbers += 1;
	}

	//正式读入和输出
	while (true)
	{
		char* end_test = fgets(line, 256, pdb_in);
		if (end_test == NULL)
		{
			break;
		}
		else if ((line[0] == 'A' && line[1] == 'T' && line[2] == 'O' && line[3] == 'M' && line[77] != 'H')
			|| (line[0] == 'H' && line[1] == 'E' && line[2] == 'T' && line[3] == 'A' && line[77] != 'H'))//垃圾原子也要如此读入
		{
			str_segment[0] = line[17];//residue name
			str_segment[1] = line[18];//residue name
			str_segment[2] = line[19];//residue name
			str_segment[3] = '_';//连接符号
			str_segment[4] = SPACE_to_UNDERSCORE(line[12]);//atom name
			str_segment[5] = SPACE_to_UNDERSCORE(line[13]);//atom name
			str_segment[6] = SPACE_to_UNDERSCORE(line[14]);//atom name
			str_segment[7] = SPACE_to_UNDERSCORE(line[15]);//atom name
			str_segment[8] = '\0';
			std::string temp_atom_name(str_segment);
			std::unordered_map<std::string, int>::iterator pointer = data_map.find(temp_atom_name);
			if (data_map.end() == pointer)
			{
				//printf("unmatched atom type %s found\n", temp_atom_name.c_str());
				for (int i = 0; i < 17; i = i + 1)
				{
					fprintf(out, "%.4f ", 0.0);
				}
				//getchar();
				//return 1;
			} else {//如果没找到对应的type
				FLOAT17 temp_float17 = data_save[pointer->second];
				for (int i = 0; i < 17; i = i + 1)
				{
					fprintf(out, "%.4f ", temp_float17.data[i]);
				}
			}
			fprintf(out, "\n");
		}
	}
	fclose(pdb_in);
	fclose(out);
	fclose(data_in);

	return 0;
}
