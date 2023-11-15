#include "DSDP_Sort.cuh"
void DSDP_SORT::Sort_Structures(const int atom_numbers, const int* atomic_number,
	const int record_numbers, const VECTOR* crd_record, const INT_FLOAT* serial_energy_list,
	const float rmsd_cutoff,const int forward_comparing_numbers, const int desired_selecting_numbers)
{
	this->atom_numbers = atom_numbers;
	selected_numbers = 0;
	//不用push_back是为了方便坐标复制
	selected_crd.resize((size_t)desired_selecting_numbers*atom_numbers);
	selected_energy.resize(desired_selecting_numbers);

	for (int frame_i=0; frame_i < record_numbers; frame_i += 1)
	{
		bool is_existing_similar_structure = false;
		for (int pose_i = selected_numbers-1; pose_i >= 0; pose_i -= 1)
		{
			float rmsd=calcualte_heavy_atom_rmsd(
				atom_numbers,
				&selected_crd[(size_t)pose_i * atom_numbers],
				&crd_record[(size_t)serial_energy_list[frame_i].id * atom_numbers],
				atomic_number);
			if (rmsd < rmsd_cutoff)
			{
				is_existing_similar_structure = true;
				break;
			}
		}
		if (is_existing_similar_structure == false)
		{
			memcpy(&selected_crd[(size_t)selected_numbers * atom_numbers], &crd_record[(size_t)serial_energy_list[frame_i].id * atom_numbers],sizeof(VECTOR)*atom_numbers);
			selected_energy[selected_numbers] = serial_energy_list[frame_i].energy;
			selected_numbers = selected_numbers + 1;
			if (selected_numbers >= desired_selecting_numbers)
			{
				break;
			}
		}
	}
}
