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
#include "../include/argparse.hpp"
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
	const VECTOR neighbor_grid_box_length = {400.f, 400.f, 400.f}; // 包住整个模拟原子的空间大小（和近邻表处理相关）
	const float cutoff = 8.f;									   // 截断
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
	unsigned int stream_numbers = 384; // 该版本每个stream就对应一个副本操作
	unsigned int search_depth = 40;	   // 每个副本尝试搜索的次数

	float box_length = 30.f;			  // another space restrain, manily because of the interpolation space limit, larger will slower(if keep interpolation precision in the same time)
	int max_record_numbers = 2000;		  // only consider top 2000 poses by energy sorting
	float rmsd_similarity_cutoff = 2.f;	  // a parameter to distinguish two different poses
	int desired_saving_pose_numbers = 50; // try to find the best 50 results to save

	double time_begin = omp_get_wtime();
	// file name
	char ligand_name[256];	 // 必须输入
	char protein_name[256];	 // 必须输入
	VECTOR box_min, box_max; // 必须输入

	char out_pdbqt_name[256] = "OUT.pdbqt";
	char out_list_name[256] = "OUT.log";
// 处理外部输入
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
	argparse::ArgumentParser program("DSDP", "1.0", argparse::default_arguments::help);
	program.add_argument("--ligand").required().help("ligand input PDBQT file").metavar("<ligand.pdbqt>");
	program.add_argument("--protein").required().help("protein input PDBQT file").metavar("<protein.pdbqt>");

	program.add_argument("--box_min").required().help("grid_box min: x y z (Angstrom)").metavar("x y z").nargs(3).scan<'g', float>();
	program.add_argument("--box_max").required().help("grid_box max: x y z (Angstrom)").metavar("x y z").nargs(3).scan<'g', float>();

	program.add_argument("--out").help("ligand poses output").metavar("<*.pdbqt>").default_value(std::string("OUT.pdbqt"));
	program.add_argument("--log").help("docking log file").metavar("<*.log>").default_value(std::string("OUT.log"));
	program.add_argument("--exhaustiveness").help("number of GPU thread (=number of copies)").metavar("N").scan<'i', int>().default_value(384);
	program.add_argument("--search_depth").help("number of searching steps for every copy").metavar("N").scan<'i', int>().default_value(40);
	program.add_argument("--top_n").help("number of desired output poses").metavar("N").scan<'i', int>().default_value(10);
	program.add_description("DSDP: Deep Site and Docking Pose\n"
							" This is the `redocking' program, run with known binding site.\n"
							" More details at https://github.com/PKUGaoGroup/DSDP");

	//  "* Cite this: J. Chem. Inf. Model. 2023, 63, 4355-4363\n"
	//  "             https://doi.org/10.1021/acs.jcim.3c00519"

	try
	{
		program.parse_args(argn, argv);
	}
	catch (const std::runtime_error &err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	sscanf(program.get<std::string>("--ligand").c_str(), "%s", ligand_name);
	sscanf(program.get<std::string>("--protein").c_str(), "%s", protein_name);
	/* Box Input */
	auto bmin = program.get<std::vector<float>>("--box_min");
	auto bmax = program.get<std::vector<float>>("--box_max");
	box_min = {bmin[0], bmin[1], bmin[2]};
	box_max = {bmax[0], bmax[1], bmax[2]};

	sscanf(program.get<std::string>("--out").c_str(), "%s", out_pdbqt_name);
	sscanf(program.get<std::string>("--log").c_str(), "%s", out_list_name);

	stream_numbers = program.get<int>("--exhaustiveness");
	search_depth = program.get<int>("--search_depth");
	desired_saving_pose_numbers = program.get<int>("--top_n");

#endif
	// 初始化
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

	// 构象初始化
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

	// 搜索
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
					float current_energy = molecule[i].vina_gpu.h_u_crd[u_freedom] / (1.f + omega * molecule[i].num_tor); // vina force field, may be for comparing different molecules with different torsion freedoms
					float probability = expf(fminf(beta * (molecule[i].vina_gpu.last_accepted_energy - current_energy), 0.f));
					if (probability > (float)rand() / RAND_MAX)
					{
						molecule[i].vina_gpu.last_accepted_energy = current_energy;
						energy_record.push_back({(int)energy_record.size(), current_energy});
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
				} // 进行接收或拒绝构象判断

				// 随机扰动
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
				cudaMemcpyAsync(molecule[i].vina_gpu.u_crd, molecule[i].vina_gpu.h_u_crd, sizeof(float) * (u_freedom + 1), cudaMemcpyHostToDevice, task[i].Get_Stream());
				Optimize_Structure_BB2_Direct_Pair_Device<<<1, 128, sizeof(float) * 23, task[i].Get_Stream()>>> // 23 is a temp array's length, just for GPU computation
					(
						molecule[i].atom_numbers, molecule[i].vina_gpu.inner_neighbor_list, cutoff,
						molecule[i].vina_gpu.atom_to_node_serial,
						molecule[i].vina_gpu.ref_crd, molecule[i].vina_gpu.d_vina_atom, molecule[i].vina_gpu.frc, &molecule[i].vina_gpu.u_crd[u_freedom],
						vgff.grid.texObj_for_kernel, border_strenth /*盒子边界强度*/,
						box_min, box_max, vgff.grid_length_inverse,
						u_freedom, molecule[i].vina_gpu.u_crd, molecule[i].vina_gpu.last_u_crd, molecule[i].vina_gpu.dU_du_crd, molecule[i].vina_gpu.last_dU_du_crd,
						molecule[i].vina_gpu.node_numbers, molecule[i].vina_gpu.node);
				cudaMemcpyAsync(molecule[i].vina_gpu.h_u_crd, molecule[i].vina_gpu.u_crd, sizeof(float) * (u_freedom + 1), cudaMemcpyDeviceToHost, task[i].Get_Stream());
				cudaMemcpyAsync(molecule[i].vina_gpu.h_vina_atom, molecule[i].vina_gpu.d_vina_atom, sizeof(VINA_ATOM) * molecule[i].atom_numbers, cudaMemcpyDeviceToHost, task[i].Get_Stream());

				task[i].Assign_Status(DSDP_TASK_STATUS::MINIMIZE_STRUCTURE);
				task[i].Record_Event();
			}
			if (search_numbers_record[i] < search_depth)
			{
				is_ok_to_break = false;
			}
		} // for 每个stream尝试
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

	sort(energy_record.begin(), energy_record.end(), cmp);
	// printf("%f\n", energy_record[0].energy);

	float energy_shift = 0.f;
	VECTOR *molecule_crd = &crd_record[(size_t)energy_record[0].id * (molecule[0].atom_numbers)];
	for (int i = 0; i < molecule[0].atom_numbers; i = i + 1)
	{
		VINA_ATOM atom_j;
		VINA_ATOM atom_i = molecule[0].vina_gpu.h_vina_atom[i];
		atom_i.crd = molecule_crd[i];
		int inner_list_start = i * molecule[0].atom_numbers;
		int inner_numbers = molecule[0].vina_gpu.h_inner_neighbor_list[inner_list_start];
		for (int k = 1; k <= inner_numbers; k = k + 1)
		{
			int j = molecule[0].vina_gpu.h_inner_neighbor_list[inner_list_start + k];
			atom_j = molecule[0].vina_gpu.h_vina_atom[j];
			atom_j.crd = molecule_crd[j];
			float2 temp = Vina_Pair_Interaction(atom_i, atom_j);
			energy_shift += temp.y;
		}
	}
	energy_shift /= (1.f + omega * molecule[0].num_tor);

	VECTOR move_vec = {-protein.move_vec.x, -protein.move_vec.y, -protein.move_vec.z};
	FILE *out_pdbqt = fopen(out_pdbqt_name, "w");
	FILE *out_list = fopen(out_list_name, "w");
	if (!out_pdbqt || !out_list)
	{
		perror("DSDP is unable to open files");
		return -1;
	}
	copy_pdbqt.Append_Frame_To_Opened_pdbqt_standard(out_pdbqt, &crd_record[(size_t)energy_record[0].id * (molecule[0].atom_numbers)], move_vec, 0, energy_record[0].energy - energy_shift);
	fprintf(out_list, "%s %f\n", ligand_name, energy_record[0].energy - energy_shift);
	// 如果要去重排序输出前n个构象
	DSDP_sort.Sort_Structures(
		molecule[0].atom_numbers, &molecule[0].atomic_number[0],
		std::min(max_record_numbers, (int)energy_record.size()), &crd_record[0], &energy_record[0],
		rmsd_similarity_cutoff, desired_saving_pose_numbers, desired_saving_pose_numbers);
	for (int i = 1; i < DSDP_sort.selected_numbers; i += 1)
	{
		copy_pdbqt.Append_Frame_To_Opened_pdbqt_standard(out_pdbqt, &DSDP_sort.selected_crd[(size_t)i * (molecule[0].atom_numbers)], move_vec, i, energy_record[i].energy - energy_shift);
		fprintf(out_list, "%s %f\n", ligand_name, energy_record[i].energy - energy_shift);
	}

	fclose(out_pdbqt);

	fclose(out_list);

	time_begin = omp_get_wtime() - time_begin;
	printf("Total time %lf s\n", time_begin);
	return 0;
}
