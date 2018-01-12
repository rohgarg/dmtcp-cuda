#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include "config.h"
#include "cuda_plugin.h"


EXTERNC cudaError_t
proxy_cudaMallocManaged(void **pointer, size_t size, unsigned int flags);

// 0.
// XXX: cudaMallocManaged is more or less the same as cudaMalloc, except
// for the additional creation of shadow pages.
EXTERNC cudaError_t
cudaMallocManaged(void **pointer, size_t size, unsigned int flags)
{
  if (!initialized)
    proxy_initialize();

#ifdef USERFAULTFD_INITIALIZED
  if (!ufd_initialized)
    userfaultfd_initialize();
#else
  if (!segvfault_initialized)
    segvfault_initialize();
#endif

  cudaError_t ret_val = proxy_cudaMallocManaged(pointer, size, flags);
  JASSERT(ret_val == cudaSuccess)\
         (ret_val).Text("Failed to create UVM region");

  // change the pointer to point the global memory (device memory)
  *pointer = create_shadow_pages(size, *pointer);

  // TODO:  Add logging

  return ret_val;
}
