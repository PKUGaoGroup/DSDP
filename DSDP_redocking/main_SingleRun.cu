#include "common.cuh"
#include "Kernel.cuh"
#include "DSDP_Task.cuh"
#include "Partial_Rigid_Small_Molecule.cuh"
#include "Neighbor_Grid.cuh"
#include "Rigid_Protein.cuh"
#include "Vina_Grid_Force_Field.cuh"
#include "Copy_pdbqt_Format.cuh"
#include "DSDP_Sort.cuh"
#include <time.h>

#define OMP_TIME
#ifdef OMP_TIME
#include <omp.h>
#endif // OMP_TIME
#define NEW_PARSE
#ifdef NEW_PARSE
#include "CLI11.hpp"
#endif
std::vector<DSDP_TASK> task;
std::vector<PARTIAL_RIGID_SMALL_MOLECULE> molecule;
RIGID_PROTEIN protein;
NEIGHBOR_GRID nl_grid;
VINA_GRID_FORCE_FIELD vgff;
COPY_pdbqt_FORMAT copy_pdbqt;
DSDP_SORT DSDP_sort;

int main(int argn, char *argv[])
{
	// for most situations, below parameters are suitable, no need for others to change
	// neighbor list related
	const VECTOR neighbor_grid_box_length = {400.f, 400.f, 400.f}; // ��ס����ģ��ԭ�ӵĿռ��С���ͽ��ڱ�������أ�
	const float cutoff = 8.f;									   // �ض�
	const float neighbor_grid_skin = 2.f;

	// interpolation list number in one dimension
	const unsigned int protein_mesh_grid_one_dimension_numbers = 100;

	// a factor to restrain small molecule in searching space
	const float border_strenth = 100.f;

	// some vina force field parameters
	const float omega = 0.292300f * 0.2f; // C_inter/(1+omega*N_rot);
	const float beta = 0.838718f;		  // 600 K, 1/(kb*T)

	// the allowed longest running time in one searching turn
	const float max_allowed_running_time = 60.f; // in sec

	// below parameters can be changed while command line input, but default may be good
	unsigned int stream_numbers = 384; // �ð汾ÿ��stream�Ͷ�Ӧһ����������
	unsigned int search_depth = 40;	   // ÿ���������������Ĵ���

	float box_length = 30.f;			  // another space restrain, manily because of the interpolation space limit, larger will slower(if keep interpolation precision in the same time)
	int max_record_numbers = 2000;		  // only consider top 2000 poses by energy sorting
	float rmsd_similarity_cutoff = 0.5f;	  // a parameter to distinguish two different poses
	int desired_saving_pose_numbers = 50; // try to find the best 50 results to save

	double time_begin = omp_get_wtime();
	// file name
	char ligand_name[256];	 // ��������
	char protein_name[256];	 // ��������
	VECTOR box_min, box_max; // ��������

	char out_pdbqt_name[256] = "OUT.pdbqt";
	char out_list_name[256] = "OUT.log";
// �����ⲿ����
#ifndef NEW_PARSE
	int necessary_input_number_record = 0;
	for (int i = 0; i < argn; i = i + 1)
	{
		// below's cmd instruction must be inputed
		if (strcmp(argv[i], "-ligand") == 0)
		{
			i += 1;
			sprintf(ligand_name, "%s", argv[i]);
			necessary_input_number_record += 1;
		}
		else if (strcmp(argv[i], "-protein") == 0)
		{
			i += 1;
			sprintf(protein_name, "%s", argv[i]);
			necessary_input_number_record += 1;
		}

		else if (strcmp(argv[i], "-box_min") == 0)
		{
			i += 1;
			sscanf(argv[i], "%f", &box_min.x);
			i += 1;
			sscanf(argv[i], "%f", &box_min.y);
			i += 1;
			sscanf(argv[i], "%f", &box_min.z);
			necessary_input_number_record += 1;
		}
		else if (strcmp(argv[i], "-box_max") == 0)
		{
			i += 1;
			sscanf(argv[i], "%f", &box_max.x);
			i += 1;
			sscanf(argv[i], "%f", &box_max.y);
			i += 1;
			sscanf(argv[i], "%f", &box_max.z);
			necessary_input_number_record += 1;
		}
		// some options
		else if (strcmp(argv[i], "-out") == 0)
		{
			i += 1;
			sprintf(out_pdbqt_name, "%s", argv[i]);
		}
		else if (strcmp(argv[i], "-log") == 0)
		{
			i += 1;
			sprintf(out_list_name, "%s", argv[i]);
		}
		else if (strcmp(argv[i], "-exhaustiveness") == 0)
		{
			i += 1;
			sscanf(argv[i], "%u", &stream_numbers);
		}
		else if (strcmp(argv[i], "-search_depth") == 0)
		{
			i += 1;
			sscanf(argv[i], "%u", &search_depth);
		}
		else if (strcmp(argv[i], "-top_n") == 0)
		{
			i += 1;
			sscanf(argv[i], "%d", &desired_saving_pose_numbers);
		}
	}
	if (necessary_input_number_record != 4)
	{
		printf("please input correct command\n");
		return -1;
		// getchar();
	}
#endif
#ifdef NEW_PARSE
	/* a new argparsing style */
	CLI::App app{"DSDP"};
	std::string ligand_string, protein_string;
	std::string out_string, log_string;
	std::vector<float> vbox_min, vbox_max;

	app.description("DSDP: Deep Site and Docking Pose\n"
					" This is the re-docking program.\n"
					" More details at https://github.com/PKUGaoGroup/DSDP");

	app.add_option("--ligand", ligand_string, "ligand input PDBQT file [REQUIRED]")
		->option_text("<pdbqt>")
		->required()
		->check(CLI::ExistingFile);
	app.add_option("--protein", protein_string, "protein input PDBQT file [REQUIRED]")
		->option_text("<pdbqt>")
		->required()
		->check(CLI::ExistingFile);

	app.add_option("--box_min", vbox_min, "grid_box min: x y z (Angstrom) [REQUIRED]")
		->option_text("x y z")
		->required()
		->expected(3);
	app.add_option("--box_max", vbox_max, "grid_box max: x y z (Angstrom) [REQUIRED]")
		->option_text("x y z")
		->required()
		->expected(3);

	app.add_option("--out", out_string, "ligand poses output [=DSDP_out.pdbqt]")
		->option_text("<pdbqt>")
		->default_val(std::string("DSDP_out.pdbqt"));

	app.add_option("--log", log_string, "log output [=DSDP_out.log]")
		->option_text("<log>")
		->default_val(std::string("DSDP_out.log"));
	app.add_option("--exhaustiveness", stream_numbers, "number of GPU threads (number of copies) [=384]")
		->default_val(384)
		->option_text("N");

	app.add_option("--search_depth", search_depth, "number of searching steps for every copy [=40]")
		->default_val(40)
		->option_text("N");
	app.add_option("--top_n", desired_saving_pose_numbers, "number of desired output poses [=10]")
		->default_val(10)
		->option_text("N");

	CLI11_PARSE(app, argn, argv);

	sscanf(ligand_string.c_str(), "%s", ligand_name);
	sscanf(protein_string.c_str(), "%s", protein_name);
	sscanf(out_string.c_str(), "%s", out_pdbqt_name);
	sscanf(log_string.c_str(), "%s", out_list_name);

	box_min = {vbox_min[0], vbox_min[1], vbox_min[2]};
	box_max = {vbox_max[0], vbox_max[1], vbox_max[2]};

#endif
	// ��ʼ��
	srand((int)time(0));
	cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleAuto);

	std::vector<VECTOR> crd_record;
	std::vector<INT_FLOAT> energy_record;

	task.resize(stream_numbers);
	for (int i = 0; i < stream_numbers; i = i + 1)
	{
		task[i].Initial();
	}
	molecule.resize(stream_numbers);
	molecule[0].Initial_From_PDBQT(ligand_name);
	for (int i = 1; i < stream_numbers; i = i + 1)
	{
		molecule[i].Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(&molecule[0]);
	}
	copy_pdbqt.Initial(ligand_name);
	int u_freedom = molecule[0].vina_gpu.u_freedom;
	nl_grid.Initial(neighbor_grid_box_length, cutoff, neighbor_grid_skin);
	vgff.Initial(protein_mesh_grid_one_dimension_numbers, cutoff);
	protein.Initial_Protein_From_PDBQT(protein_name, neighbor_grid_box_length);
	nl_grid.gpu.Put_Atom_Into_Grid_Bucket(protein.atom_numbers, &protein.crd[0]);
	box_min.x += protein.move_vec.x;
	box_min.y += protein.move_vec.y;
	box_min.z += protein.move_vec.z;
	box_max.x += protein.move_vec.x;
	box_max.y += protein.move_vec.y;
	box_max.z += protein.move_vec.z;
	vgff.grid.Calculate_Protein_Potential_Grid(
		box_min, box_length,
		protein.atom_numbers, protein.d_vina_atom,
		nl_grid.grid_length_inverse, nl_grid.grid_dimension, nl_grid.gpu.neighbor_grid_bucket);

	// �����ʼ��
	for (int i = 0; i < stream_numbers; i = i + 1)
	{
		VECTOR rand_vec = {(float)0.5f * rand() / RAND_MAX + 0.25f, (float)0.5f * rand() / RAND_MAX + 0.25f, (float)0.5f * rand() / RAND_MAX + 0.25f};
		for (int j = 0; j < u_freedom - 6; j = j + 1)
		{
			molecule[i].vina_gpu.h_u_crd[j] = 2.f * 3.141592654f * rand() / RAND_MAX;
		}
		VECTOR rand_angle = unifom_rand_Euler_angles();
		molecule[i].vina_gpu.h_u_crd[u_freedom - 3] = rand_angle.z;
		molecule[i].vina_gpu.h_u_crd[u_freedom - 2] = rand_angle.y;
		molecule[i].vina_gpu.h_u_crd[u_freedom - 1] = rand_angle.x;

		molecule[i].vina_gpu.h_u_crd[u_freedom - 6] = (box_min.x + rand_vec.x * (box_max.x - box_min.x));
		molecule[i].vina_gpu.h_u_crd[u_freedom - 5] = (box_min.y + rand_vec.y * (box_max.y - box_min.y));
		molecule[i].vina_gpu.h_u_crd[u_freedom - 4] = (box_min.z + rand_vec.z * (box_max.z - box_min.z));

		molecule[i].vina_gpu.last_accepted_energy = 1000.f;
		task[i].Assign_Status(DSDP_TASK_STATUS::EMPTY);
		memcpy(molecule[i].vina_gpu.h_last_accepted_u_crd, molecule[i].vina_gpu.h_u_crd, sizeof(float) * u_freedom);
	}

	// ����
#ifdef OMP_TIME
	double time_start = omp_get_wtime();
#endif // OMP_TIME
	// std::vector<VECTOR>crd_record;
	// std::vector<INT_FLOAT>energy_record;
	std::vector<int> search_numbers_record(stream_numbers, 0);
	cudaDeviceSynchronize();
	while (true)
	{
		bool is_ok_to_break = true;
		for (int i = 0; i < stream_numbers; i = i + 1)
		{
			if (task[i].Is_empty())
			{
				if (task[i].Get_Status() == DSDP_TASK_STATUS::MINIMIZE_STRUCTURE)
				{
					// vina force field, may be for comparing different molecules with different torsion freedoms
					float current_energy = molecule[i].vina_gpu.h_u_crd[u_freedom]; // / (1.f + omega * molecule[i].num_tor);
					// intra energy
					float current_intra_energy = molecule[i].vina_gpu.h_u_crd[u_freedom + 1];// / (1.f + omega * molecule[i].num_tor);
					float probability = expf(fminf(beta * (molecule[i].vina_gpu.last_accepted_energy - current_energy), 0.f));
					if (probability > (float)rand() / RAND_MAX)
					{
						molecule[i].vina_gpu.last_accepted_energy = current_energy;
						energy_record.push_back({(int)energy_record.size(), current_energy - current_intra_energy});
						for (int j = 0; j < molecule[i].atom_numbers; j = j + 1)
						{
							crd_record.push_back(molecule[i].vina_gpu.h_vina_atom[j].crd);
						}
						memcpy(molecule[i].vina_gpu.h_last_accepted_u_crd, molecule[i].vina_gpu.h_u_crd, sizeof(float) * u_freedom);
					}
					else
					{
						memcpy(molecule[i].vina_gpu.h_u_crd, molecule[i].vina_gpu.h_last_accepted_u_crd, sizeof(float) * u_freedom);
					}
				} // ���н��ջ�ܾ������ж�

				// ����Ŷ�
				int rand_int = rand() % u_freedom;
				if (rand_int < u_freedom - 3)
				{
					if (rand_int < u_freedom - 6)
					{
						molecule[i].vina_gpu.h_u_crd[rand_int] = 2.f * 3.141592654f * ((float)rand() / RAND_MAX);
					}
					else
					{
						molecule[i].vina_gpu.h_u_crd[rand_int] += 1.f * (2.f * ((float)rand() / RAND_MAX) - 1.f);
					}
				}
				else
				{
					VECTOR rand_angle = unifom_rand_Euler_angles();
					molecule[i].vina_gpu.h_u_crd[u_freedom - 3] = rand_angle.z;
					molecule[i].vina_gpu.h_u_crd[u_freedom - 2] = rand_angle.y;
					molecule[i].vina_gpu.h_u_crd[u_freedom - 1] = rand_angle.x;
				}

				search_numbers_record[i] += 1;
				cudaMemcpyAsync(molecule[i].vina_gpu.u_crd, molecule[i].vina_gpu.h_u_crd, sizeof(float) * (u_freedom + 2), cudaMemcpyHostToDevice, task[i].Get_Stream());
				Optimize_Structure_BB2_Direct_Pair_Device<<<1, 128, sizeof(float) * 23, task[i].Get_Stream()>>> // 23 is a temp array's length, just for GPU computation
					(
						molecule[i].atom_numbers, molecule[i].vina_gpu.inner_neighbor_list, cutoff,
						molecule[i].vina_gpu.atom_to_node_serial,
						molecule[i].vina_gpu.ref_crd, molecule[i].vina_gpu.d_vina_atom, molecule[i].vina_gpu.frc, &molecule[i].vina_gpu.u_crd[u_freedom],
						vgff.grid.texObj_for_kernel, border_strenth /*���ӱ߽�ǿ��*/,
						box_min, box_max, vgff.grid_length_inverse,
						u_freedom, molecule[i].vina_gpu.u_crd, molecule[i].vina_gpu.last_u_crd, molecule[i].vina_gpu.dU_du_crd, molecule[i].vina_gpu.last_dU_du_crd,
						molecule[i].vina_gpu.node_numbers, molecule[i].vina_gpu.node);
				cudaMemcpyAsync(molecule[i].vina_gpu.h_u_crd, molecule[i].vina_gpu.u_crd, sizeof(float) * (u_freedom + 2), cudaMemcpyDeviceToHost, task[i].Get_Stream());
				cudaMemcpyAsync(molecule[i].vina_gpu.h_vina_atom, molecule[i].vina_gpu.d_vina_atom, sizeof(VINA_ATOM) * molecule[i].atom_numbers, cudaMemcpyDeviceToHost, task[i].Get_Stream());

				task[i].Assign_Status(DSDP_TASK_STATUS::MINIMIZE_STRUCTURE);
				task[i].Record_Event();
			}
			if (search_numbers_record[i] < search_depth)
			{
				is_ok_to_break = false;
			}
		} // for ÿ��stream����
#ifdef OMP_TIME
		if (omp_get_wtime() - time_start > max_allowed_running_time)
		{
			is_ok_to_break = true;
		}
#endif // OMP_TIME
		if (is_ok_to_break)
		{
			break;
		}
	}
	cudaDeviceSynchronize();
#ifdef OMP_TIME

	time_start = omp_get_wtime() - time_start;

	printf("%s\n", ligand_name);
#endif // OMP_TIME


//first sort
	sort(energy_record.begin(), energy_record.end(), cmp);

//second search
//...
//



//second sort
	sort(energy_record.begin(), energy_record.end(), cmp);
	DSDP_sort.Sort_Structures(
			molecule[0].atom_numbers, &molecule[0].atomic_number[0],
			std::min(max_record_numbers, (int)energy_record.size()), &crd_record[0], &energy_record[0],
			rmsd_similarity_cutoff, 20, 20);

	//out
	FILE *out_pdbqt = fopen(out_pdbqt_name, "w");
	FILE *out_list = fopen(out_list_name, "w");
	if (!out_pdbqt || !out_list)
	{
		perror("DSDP is unable to open files");
		return -1;
	}

	//third sort but with refine
	//omp_set_num_threads(8);
	int refine_structure_numbers=std::min(20,DSDP_sort.selected_numbers);
	energy_record.resize(refine_structure_numbers);
	crd_record.resize(refine_structure_numbers*molecule[0].atom_numbers);
	#pragma omp parallel for
	for(int i=0;i<refine_structure_numbers;i=i+1)
	{

		molecule[i].Refresh_origin_crd(& DSDP_sort.selected_crd[(size_t)i*molecule[i].atom_numbers]);
		molecule[i].Build_Neighbor_List(15.f,protein.atom_numbers,&protein.crd[0],&protein.atomic_number[0]);
		float score=molecule[i].Refine_Structure(&protein.vina_atom[0],&protein.crd[0]);
		VECTOR move_vec = {molecule[i].move_vec.x-protein.move_vec.x,molecule[i].move_vec.y -protein.move_vec.y,molecule[i].move_vec.z -protein.move_vec.z};
		for (int j = 0; j < molecule[i].atom_numbers; j = j + 1)
		{
			crd_record[(size_t)i*molecule[i].atom_numbers+j]={molecule[i].crd[j].x+move_vec.x,molecule[i].crd[j].y+move_vec.y,molecule[i].crd[j].z+move_vec.z};
		}
		energy_record[(size_t)i]={i,score};
	}
	sort(energy_record.begin(), energy_record.end(), cmp);

	for (int i=0;i<desired_saving_pose_numbers;i+=1)
	{
	copy_pdbqt.Append_Frame_To_Opened_pdbqt_standard(out_pdbqt, &crd_record[(size_t)energy_record[i].id*molecule[0].atom_numbers], {0.f,0.f,0.f}, i, energy_record[i].energy/(1.f + omega * molecule[0].num_tor));
	fprintf(out_list, "%s %f\n", ligand_name, energy_record[i].energy/(1.f + omega * molecule[0].num_tor));
	}

	fclose(out_pdbqt);

	fclose(out_list);

	time_begin = omp_get_wtime() - time_begin;
	printf("Total time %lf s\n", time_begin);
	return 0;
}
