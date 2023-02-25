#ifndef DSDP_TASK_CUH
#define DSDP_TASK_CUH
#include "common.cuh"

enum DSDP_TASK_STATUS
{
	EMPTY,
	MINIMIZE_STRUCTURE,
	CALCULATE_ENERGY_AND_GRAD,
	NOT_INITIALIZED
};
struct DSDP_TASK
{
public:
	void Initial();
	bool Is_empty();
	cudaStream_t Get_Stream();
	void Record_Event();
	void Assign_Status(const DSDP_TASK_STATUS status);
	DSDP_TASK_STATUS Get_Status();
	void Clear();
private:
	bool is_initialized = false;
	cudaStream_t stream;
	cudaEvent_t event;
	DSDP_TASK_STATUS status= EMPTY;
};
#endif 