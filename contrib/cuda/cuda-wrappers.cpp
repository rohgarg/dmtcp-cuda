#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <limits.h>

#include "config.h"
#include "cuda_plugin.h"

struct MallocRegion {
  void *addr;
  void *host_addr;
  size_t len;
};

EXTERNC cudaError_t
proxy_cudaMallocManagedMemcpy(void *dst, void *src,
                              size_t size, cudaMemcpyKind kind);
EXTERNC cudaError_t
proxy_cudaMallocManaged(void **pointer, size_t size, unsigned int flags);

static dmtcp::vector<MallocRegion>&
allMallocRegions()
{
  static dmtcp::vector<MallocRegion> *instance = NULL;
  if (instance == NULL) {
    void *buffer = JALLOC_MALLOC(1024 * 1024);
    instance = new (buffer)dmtcp::vector<MallocRegion>();
  }
  return *instance;
}

void
create_copy_of_data_on_host(CudaCallLog_t *l)
{
  enum cuda_op op;
  memcpy(&op, l->fncargs, sizeof op);
  if (op == OP_cudaMalloc) {
    size_t len;
    void *devPtr;
    memcpy(&len, l->fncargs + sizeof op, sizeof len);
    memcpy(&devPtr, l->results, sizeof devPtr);
    void *page = mmap(NULL, len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    JASSERT(page != MAP_FAILED)(JASSERT_ERRNO);
    l->host_addr = page;
    cudaMemcpy(page, devPtr, len, cudaMemcpyDeviceToHost);
  } else if (op == OP_proxy_cudaMallocManaged) {
    size_t len;
    void *devPtr;
    memcpy(&len, l->fncargs + sizeof op, sizeof len);
    memcpy(&devPtr, l->results, sizeof devPtr);
    // NOTE: devPtr on the proxy and pointer to the corresponding
    // shadow page in the master process have the same virtual
    // address
    l->host_addr = devPtr;
    ShadowRegion *r = getShadowRegionForAddr(devPtr);
    JASSERT(r != NULL)(devPtr);
    // We don't copy data over dirty shadow pages.
    if (!r->dirty) {
      // NOTE: The perms on the shadow pages were set to RW earlier
      // via a call to unregister_all_pages(). The perms will be restored
      // later via a call to register_all_pages() in the resume barrier.
      cudaError_t ret_val =
           proxy_cudaMallocManagedMemcpy(devPtr, devPtr, len,
                                         cudaMemcpyDeviceToHost);
      JASSERT(ret_val == cudaSuccess)(ret_val)
              .Text("Failed to copy UVM data to host");
    }
  } else {
    JWARNING(false)(op).Text("Unknown op code; no data to copy?");
  }
}

void
send_saved_data_to_device(CudaCallLog_t *l)
{
  enum cuda_op op;
  memcpy(&op, l->fncargs, sizeof op);
  if (op == OP_cudaMalloc) {
    size_t len;
    void *devPtr;
    memcpy(&len, l->fncargs + sizeof op, sizeof len);
    memcpy(&devPtr, l->results, sizeof devPtr);
    void *newDevPtr = NULL;
    cudaError_t ret = cudaMalloc(&newDevPtr, len);
    JASSERT(ret == cudaSuccess && newDevPtr == devPtr);
    JASSERT(l->host_addr != NULL);
    cudaMemcpy(devPtr, l->host_addr, len, cudaMemcpyHostToDevice);
  } else if (op == OP_proxy_cudaMallocManaged) {
    size_t len;
    unsigned int flags;
    void *devPtr;
    memcpy(&len, l->fncargs + sizeof op, sizeof len);
    memcpy(&flags, l->fncargs + sizeof op + sizeof len, sizeof flags);
    memcpy(&devPtr, l->results, sizeof devPtr);

    void *newDevPtr = NULL;
    cudaError_t ret_val = proxy_cudaMallocManaged(&newDevPtr, len, flags);
    JASSERT(ret_val == cudaSuccess)\
           (ret_val).Text("Failed to create UVM region");
    JASSERT(newDevPtr == devPtr && l->host_addr == devPtr);
    JASSERT(l->host_addr != NULL);
    ShadowRegion *r = getShadowRegionForAddr(devPtr);
    JASSERT(r != NULL)(devPtr);
    // We don't copy data over dirty shadow pages.
    if (!r->dirty) {
      ret_val = proxy_cudaMallocManagedMemcpy(devPtr, devPtr, len,
                                              cudaMemcpyHostToDevice);
      JASSERT(ret_val == cudaSuccess)(ret_val)
              .Text("Failed to send UVM dirty pages");
    }
  } else {
    JASSERT(false)(op).Text("Replaying unknown op code");
  }
}

void
copy_data_to_host()
{
  logs_read_and_apply(create_copy_of_data_on_host);
}

void
copy_data_to_device()
{
  logs_read_and_apply(send_saved_data_to_device);
}

EXTERNC CUresult
cuInit(unsigned int Flags)
{
  JWARNING(false).Text("Missing wrapper?");
  int dummy = 0;
  while (!dummy);
}

void
print_stats()
{
#ifdef STATS
  int i;
  printf("Wrappers: CUDA call cost:\n");
  for (i = 0; i < OP_LAST_FNC; i++) {
    if (costs[i].count > 0) {
       printf("%d, %d, %llu, %llu\n", i, costs[i].count,
              costs[i].totalTime, costs[i].cudaCallCost);
    }
  }
#endif // ifdef STATS
}


// Auto-generated code
# include "python-auto-generate/cudawrappers.icpp"
