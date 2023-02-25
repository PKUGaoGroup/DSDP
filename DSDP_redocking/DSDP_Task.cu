#include "DSDP_Task.cuh"
void DSDP_TASK::Initial()
{
	if (!is_initialized)
	{
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);//cudaStreamDefault cudaStreamNonBlocking
		cudaEventCreateWithFlags(&event, cudaEventDisableTiming);//初始的时候cudaEventQuery返回cudaSuccess
		status = DSDP_TASK_STATUS::EMPTY;
		is_initialized = true;
	}
}
bool DSDP_TASK::Is_empty()
{
	if (cudaEventQuery(event) == cudaSuccess)
	{
		return true;
	}
	else
	{
		return false;
	}
}
cudaStream_t DSDP_TASK::Get_Stream()
{
	return stream;
}
void DSDP_TASK::Record_Event()
{
	cudaEventRecord(event, stream);
}
void DSDP_TASK::Clear()
{
	if (is_initialized)
	{
		cudaStreamDestroy(stream);
		cudaEventDestroy(event);
		status = DSDP_TASK_STATUS::NOT_INITIALIZED;
		is_initialized = false;
	}
}
void DSDP_TASK::Assign_Status(const DSDP_TASK_STATUS status)
{
	this->status = status;
}

DSDP_TASK_STATUS DSDP_TASK::Get_Status()
{
	return status;
}
