#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include "config.h"
#include "cuda_plugin.h"


// 0.
EXTERNC cudaError_t
cudaMallocManaged(void **pointer, size_t size, unsigned int flags)
{
  if (!initialized)
    proxy_initialize();

  if (!ufd_initialized)
    userfaultfd_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocManaged;
  strce_to_send.syscall_type.cuda_malloc.pointer = *pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;
  // TODO: Add field for flags

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  // change the pointer to point the global memory (device memory)
  *pointer = create_shadow_pages(size, &rcvd_strce);

  // record this system call to the log file
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaMallocManaged;
  strce_to_send.syscall_type.cuda_malloc.pointer = pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;

  log_append(strce_to_send);

  return ret_val;
}
