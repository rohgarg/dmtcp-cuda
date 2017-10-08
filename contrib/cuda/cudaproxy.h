#ifndef _CUDA_PROXY_H_
#define _CUDA_PROXY_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define False 0
#define True 1

enum cuda_syscalls
{
  CudaMalloc, CudaFree, CudaMallocArray, CudaFreeArray, 
  CudaMemcpy, CudaHostAlloc, CudaConfigureCall, CudaSetupArgument, 
  CudaLaunch
};

// the structure for all our cuda system calls
// so far it's for the following functions
// cudaMalloc, cudaMallocArray, cudaFree, and cudaMemcpy
typedef struct 
{
  enum cuda_syscalls op;
  union
  {
    struct 
    {
      // the structure takes a deferenced pointer
      // Since it's the proxy that calls cudaMalloc() 
      // &pointer, (void **), will then be passed to cudaMalloc.
      void *pointer;
      size_t size;
    } cuda_malloc;

    struct 
    {
      void *pointer;
    } cuda_free;

    struct
    {
      // the structure takes a deferenced pointer
      // Since it's the proxy that calls cudaMallocArray() 
      // &array (cudaArray **) will then be passed to cudaMalloc.
      struct cudaArray *array;
      // it's the proxy that will pass &desc to cudaMallocArray()
      struct cudaChannelFormatDesc desc; 
      size_t width;
      size_t height;
      unsigned int flags;
    } cuda_malloc_array;

    struct 
    {
      struct cudaArray *array;
    } cuda_free_array;

    struct
    {
      void *destination;
      const void *source;
      size_t size;
      enum cudaMemcpyKind direction;
    } cuda_memcpy;
  
//    struct
//    {
//      // master and proxy will have different pointer to the shared memory
//      // we only include pHost in this structure for record purpose
//      void *pHost;
//      size_t size;
//      unsigned int flags;
//    } cuda_host_alloc
    
    struct
    {
      int gridDim[3];    // to mimic dim3
      int blockDim[3];   // to mimic dim3
      size_t sharedMem;
      cudaStream_t stream;
    } cuda_configure_call;

    struct
    {
      const void *arg; // it's used for record. 
      size_t size;
      size_t offset;
    } cuda_setup_argument;


    struct
    {
      int shmId;
    } cuda_launch;

    struct
    {
      const void *func;
    } cuda_launch_record;

  }syscall_type;
  const void *payload;
  size_t payload_size;
} cudaSyscallStructure;

#endif // _CUDA_PROXY_H_
