#ifndef __CUDA_PLUGIN_H
#define __CUDA_PLUGIN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef STANDALONE
#include <stdint.h>

#include "jassert.h"
#include "dmtcp.h"
#include "dmtcp_dlsym.h"

#define DEBUG_SIGNATURE "[CUDA Plugin]"
#ifdef CUDA_PLUGIN_DEBUG
# define DPRINTF(fmt, ...) \
  do { fprintf(stderr, DEBUG_SIGNATURE fmt, ## __VA_ARGS__); } while (0)
#else // ifdef CUDA_PLUGIN_DEBUG
# define DPRINTF(fmt, ...) \
  do {} while (0)
#endif // ifdef CUDA_PLUGIN_DEBUG


#define   _real_fork      NEXT_FNC_DEFAULT(fork)

#define   _real_cudaMalloc      NEXT_FNC_DEFAULT(cudaMalloc)
#define   _real_cudaMemcpy      NEXT_FNC_DEFAULT(cudaMemcpy)
#define   _real_dlopen          NEXT_FNC_DEFAULT(dlopen)
#define   _real_dlclose         NEXT_FNC_DEFAULT(dlclose)
#define   _real_dlsym           NEXT_FNC_DEFAULT(dlsym)
#define   _real_cuLaunchKernel  NEXT_FNC_DEFAULT(cuLaunchKernel)

#define   _real_cudaConfigureCall     NEXT_FNC_DEFAULT(cudaConfigureCall)
#define   _real_cudaLaunch            NEXT_FNC_DEFAULT(cudaLaunch)
#define   _real_cudaFuncGetAttributes NEXT_FNC_DEFAULT(cudaFuncGetAttributes)
#define   _real_cudaSetupArgument     NEXT_FNC_DEFAULT(cudaSetupArgument)
#define   _real_cudaLaunchKernel      NEXT_FNC_DEFAULT(cudaLaunchKernel)

#define True    1
#define False   0

#define SKTNAME "proxy"
#define LOGFILE "cudaSysCallsLog"

#define SHMSIZE 128

extern int initialized;
extern int ufd_initialized;
extern void *shmaddr;
extern int logFd;
extern int page_size;

// master socket
extern int skt_master;

// proxy address
extern struct sockaddr_un sa_proxy;
#endif // ifndef STANDALONE

enum cuda_syscalls
{
  CudaMalloc, CudaFree, CudaMallocArray, CudaFreeArray,
  CudaMallocManaged, CudaMallocManagedMemcpy,
  CudaMemcpy, CudaHostAlloc, CudaConfigureCall, CudaSetupArgument,
  CudaLaunch,
  CudaDeviceSync,
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

    struct
    {
      void *destination;
      const void *source;
      size_t size;
      enum cudaMemcpyKind direction;
    } cuda_managed_memcpy;

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

#ifndef STANDALONE

static inline void*
getAlignedAddress(uintptr_t ptr, size_t alignment)
{
  const size_t mask = alignment - 1;
  return (void *) (ptr & ~mask);
}

void proxy_initialize(void);
void send_recv(int fd, cudaSyscallStructure *strce_to_send,
              cudaSyscallStructure *rcvd_strce, cudaError_t *ret_val);
void log_append(cudaSyscallStructure record);
bool log_read(cudaSyscallStructure *record);

void userfaultfd_initialize(void);
void reset_uffd(void);
void* create_shadow_pages(size_t size, cudaSyscallStructure *remoteInfo = NULL);
void unregister_all_pages();
void register_all_pages();

dmtcp::map<void*, void*>& shadowPageMap();

#endif // ifndef STANDALONE

#endif // ifndef  __CUDA_PLUGIN_H
