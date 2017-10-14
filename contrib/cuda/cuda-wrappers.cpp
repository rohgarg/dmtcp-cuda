#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include "config.h"
#include "cuda_plugin.h"

// 1.
EXTERNC cudaError_t
cudaMalloc(void **pointer, size_t size)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMalloc;
  strce_to_send.syscall_type.cuda_malloc.pointer = *pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  void *record_pointer = *pointer;

  // change the pointer to point the global memory (device memory)
  *pointer = rcvd_strce.syscall_type.cuda_malloc.pointer;

  // record this system call to the log file
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaMalloc;
  strce_to_send.syscall_type.cuda_malloc.pointer = pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;

  log_append(strce_to_send);

  return ret_val;
}

// 2.
EXTERNC cudaError_t
cudaFree(void *pointer)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaFree;
  strce_to_send.syscall_type.cuda_free.pointer = pointer;


  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaFree;
  strce_to_send.syscall_type.cuda_free.pointer = pointer;

  log_append(strce_to_send);

  return ret_val;
}

// 3.
EXTERNC cudaError_t
cudaMemcpy(void *pointer1, const void *pointer2, size_t size,
           enum cudaMemcpyKind direction)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val = cudaSuccess;

  strce_to_send.op = CudaMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = pointer1;
  strce_to_send.syscall_type.cuda_memcpy.source = (void*)pointer2;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = direction;

  switch(direction)
  {
    JTRACE("cudaMemcpy(): lib");

    case cudaMemcpyHostToHost:
    {
      memcpy(pointer1, pointer2, size);
      return ret_val;
    }


    case cudaMemcpyHostToDevice:
      strce_to_send.payload = pointer2;
      strce_to_send.payload_size = size;
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
      break;

    case cudaMemcpyDeviceToHost:
      // send the structure
      JASSERT(write(skt_master, &strce_to_send, sizeof(strce_to_send)) != -1)
             (JASSERT_ERRNO);

      // get the payload: part of the GPU computation actually
      JASSERT(read(skt_master, pointer1, size) != -1)(JASSERT_ERRNO);

      // receive the result
      memset(&ret_val, 0, sizeof(ret_val));

      JASSERT(read(skt_master, &ret_val, sizeof(ret_val)) != -1)(JASSERT_ERRNO);

      JASSERT(ret_val == cudaSuccess).Text("cudaMemcpy failed");

      // get the structure back
      memset(&rcvd_strce, 0, sizeof(rcvd_strce));
      JASSERT(read(skt_master, &rcvd_strce, sizeof(rcvd_strce)) != -1)
             (JASSERT_ERRNO);
      break;

    case cudaMemcpyDeviceToDevice:
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

      break;

    default:
      JASSERT(false)(direction).Text("Unknown direction for memcpy");
  }

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = pointer1;
  strce_to_send.syscall_type.cuda_memcpy.source = pointer2;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = direction;

  log_append(strce_to_send);
  return ret_val;
}

// 4.
EXTERNC cudaError_t
cudaMallocArray(struct cudaArray **array,
                const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaMallocArray;
  strce_to_send.syscall_type.cuda_malloc_array.array = *array;
  strce_to_send.syscall_type.cuda_malloc_array.desc = *desc;
  strce_to_send.syscall_type.cuda_malloc_array.width = width;
  strce_to_send.syscall_type.cuda_malloc_array.height = height;
  strce_to_send.syscall_type.cuda_malloc_array.flags = flags;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  struct cudaArray *record_array = *array;

  // change the pointer to point the global memory (device memory)
  *array = rcvd_strce.syscall_type.cuda_malloc_array.array;

  // record this system call to the log file
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaMallocArray;
  strce_to_send.syscall_type.cuda_malloc_array.array = record_array;
  strce_to_send.syscall_type.cuda_malloc_array.desc = *desc;
  strce_to_send.syscall_type.cuda_malloc_array.width = width;
  strce_to_send.syscall_type.cuda_malloc_array.height = height;
  strce_to_send.syscall_type.cuda_malloc_array.flags = flags;

  log_append(strce_to_send);

  return ret_val;
}

// 5.
EXTERNC cudaError_t
cudaFreeArray(struct cudaArray *array)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaFreeArray;
  strce_to_send.syscall_type.cuda_free_array.array = array;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaFreeArray;
  strce_to_send.syscall_type.cuda_free_array.array = array;

  log_append(strce_to_send);

  return ret_val;
}


#if 0
// 6.
void cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  int ret_val;

  strce_to_send.op = CudaHostAlloc;
  strce_to_send.syscall_type.cuda_host_alloc.pHost = *pHost;
  strce_to_send.syscall_type.cuda_host_alloc.size = size;
  strce_to_send.syscall_type.cuda_host_alloc.flags = flags;

  //
  send_recv(skt_master, &strce_to_send, &rcv_strce, &ret_val);

  // receive shmid
  int shmid;
  if (read(skt_master, &shmid, sizeof(int)) == -1)
  {
    perror("read()");
    exit(EXIT_SUCCESS);
  }

  // attach the shared memory
  void *addr;
  if ((addr = shmat(shmid, NULL, 0)) == -1)
  {
    perror("shmat()");
    exit(EXIT_FAILURE);
  }
  //

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaHostAlloc;
  strce_to_send.syscall_type.cuda_host_alloc.pHost = *pHost;
  strce_to_send.syscall_type.cuda_host_alloc.size = size;
  strce_to_send.syscall_type.cuda_host_alloc.flags = flags;

  // change pHost to point to the shared memory
  *pHost = addr;

  // record this function call
  log_append(strce_to_send);
}
#endif


//
EXTERNC cudaError_t
cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                  size_t sharedMem, cudaStream_t stream)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaConfigureCall;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[0] = gridDim.x;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[1] = gridDim.y;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[2] = gridDim.z;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[0] = blockDim.x;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[1] = blockDim.y;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[2] = blockDim.z;
  strce_to_send.syscall_type.cuda_configure_call.sharedMem = sharedMem;
  strce_to_send.syscall_type.cuda_configure_call.stream = stream;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  return ret_val;
}

//
EXTERNC cudaError_t
cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
  if (!initialized)
    proxy_initialize();

#if USE_SHM
  memcpy(shmaddr, arg, size);
#endif

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaSetupArgument;
  strce_to_send.syscall_type.cuda_setup_argument.size = size;
  strce_to_send.syscall_type.cuda_setup_argument.offset = offset;
  strce_to_send.payload = arg;
  strce_to_send.payload_size = size;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaSetupArgument;
  strce_to_send.syscall_type.cuda_setup_argument.arg = arg;
  strce_to_send.syscall_type.cuda_setup_argument.size = size;
  strce_to_send.syscall_type.cuda_setup_argument.offset = offset;

  log_append(strce_to_send);

  return ret_val;
}

//
EXTERNC cudaError_t
cudaLaunch(const void *func)
{
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  strce_to_send.op = CudaLaunch;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaLaunch;

  log_append(strce_to_send);

  return ret_val;
}
