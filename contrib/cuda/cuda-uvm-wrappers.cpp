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

#ifdef CUDA_PASCAL
  // Note: UVM regions on Pascal can share a page. This can
  // cause problems for us, since the later code assumes one
  // UVM region per page. Here, we force every allocation to be
  // at least a page size to avoid this problem. This
  // restriction/hack can be fixed later. FIXME
  int npages = (size % page_size == 0) ?
               (size / page_size) : (size / page_size + 1);
  size = npages * page_size;
  JASSERT(size % page_size == 0)(size);
#endif

  cudaError_t ret_val = proxy_cudaMallocManaged(pointer, size, flags);
  JASSERT(ret_val == cudaSuccess)\
         (ret_val).Text("Failed to create UVM region");

  if (*pointer == NULL) {
    JTRACE("UVM pointer is NULL. Perhaps the size was 0?")
          (*pointer)(size)(flags);
    return ret_val;
  }

  // change the pointer to point the global memory (device memory)
  *pointer = create_shadow_pages(size, *pointer);

  // TODO:  Add logging

  return ret_val;
}
