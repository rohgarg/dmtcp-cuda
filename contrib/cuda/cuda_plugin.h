#ifndef __CUDA_PLUGIN_H
#define __CUDA_PLUGIN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <sys/time.h>

// The following declarations, definitions, etc. are common
// to the DMTCP CUDA plugin and the CUDA proxy.

#include "python-auto-generate/cuda_plugin.h"

#ifdef STATS

struct CallCost
{
  uint64_t count;
  uint64_t totalTime;
  uint64_t cudaCallCost;
};

# define FNC_START_TIME(op) \
    costs[op].count++; \
    struct timeval start = {0}; \
    struct timeval end = {0}; \
    gettimeofday(&start, NULL);

# define FNC_END_TIME(op) \
    gettimeofday(&end, NULL); \
    struct timeval res; \
    timersub(&end, &start, &res); \
    costs[op].totalTime += res.tv_sec * 1e6 + res.tv_usec; \


# define CUDA_CALL_START_TIME(myop) \
  { \
    struct timespec spec; \
    clock_gettime(CLOCK_REALTIME, &spec); \
    uint64_t sec = spec.tv_sec * 1e9 + spec.tv_nsec;

# define CUDA_CALL_END_TIME(myop) \
    clock_gettime(CLOCK_REALTIME, &spec); \
    sec = spec.tv_sec * 1e9 + spec.tv_nsec - sec; \
    costs[(myop)].cudaCallCost += sec; \
  }

extern struct CallCost costs[OP_LAST_FNC];

extern uint64_t totalTimeInUvmCopy;
extern std::map<uint64_t, uint64_t> uvmReadReqMap;
extern std::map<uint64_t, uint64_t> uvmWriteReqMap;

#else

# define FNC_START_TIME(op)
# define FNC_END_TIME(op)
# define CUDA_CALL_START_TIME(myop)
# define CUDA_CALL_END_TIME(myop)

#endif // ifdef STATS

extern void print_stats();

#ifndef STANDALONE

// The following declarations, definitions, etc. are only
// valid for the DMTCP CUDA plugin

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
extern int segvfault_initialized;
extern void *shmaddr;
extern int logFd;
extern int page_size;

extern bool haveDirtyPages;

// master socket
extern int skt_master;

// proxy address
extern struct sockaddr_un sa_proxy;

extern bool enableCudaCallLogging;
#endif // ifndef STANDALONE

enum cuda_syscalls
{
  CudaMalloc,
  CudaMallocPitch,
  CudaMallocManaged,
  CudaMallocManagedMemcpy,
  CudaFree,
  CudaMallocArray,
  CudaFreeArray,
  CudaMemcpy,
  CudaHostAlloc,
  CudaConfigureCall,
  CudaSetupArgument,
  CudaLaunch,
  CudaThreadSync,
  CudaGetLastError,
  CudaGetErrorString,
  CudaMemcpy2D,
  CudaDeviceReset,
  CudaMemcpyToSymbol,
  CudaCreateChannelDesc,
  CudaBindTexture2D,
  /*-- Added to support HPGMG-CUDA */
  CudaBindTexture,
  CudaCreateTextureObject,
  CudaPeekAtLastError,
  CudaProfilerStart,
  CudaProfilerStop,
  CudaStreamSynchronize,
  CudaUnbindTexture,
  CudaDestroyTextureObject,
  CudaEventDestroy,
  CudaEventQuery,
  CudaFreeHost,
  CudaDeviceCanAccessPeer,
  CudaDeviceGetAttribute,
  CudaDeviceSetCacheConfig,
  CudaDeviceSetSharedMemConfig,
  CudaDeviceSynchronize,
  CudaEventCreateWithFlags,
  CudaEventRecord,
  CudaFuncGetAttributes,
  CudaGetDevice,
  CudaGetDeviceCount,
  CudaGetDeviceProperties,
  CudaMallocHost, // to be revisited
  CudaMemcpyAsync, // to be revisited
  CudaMemset,
  CudaMemsetAsync, // to be revisited
  CudaSetDevice, // here
  CudaPointerGetAttributes,
  CudaOccupancyMaxActiveBlocksPerMultiprocessor
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
      const void *func_addr;
    } cuda_launch;

//    struct
//    {
//      const void *func;
//    } cuda_launch_record;
    struct
    {
      cudaError_t errorCode;
      // return value non-trivial (i.e. other than cudaError_t)
      char *error_string;
      size_t size;
    } cuda_get_error_string;

    struct
    {
      // the structure takes a deferenced pointer
      // Since it's the proxy that calls cudaMallocPitch()
      // &devPtr, (void *), will then be passed to cudaMallocPitch.
      void* devPtr;
      // original argument is "*pitch" but this makes information
      // exchange between wrapper and proxy easier.
      size_t pitch;
      size_t *pitchp;
      size_t width;
      size_t height;
    } cuda_malloc_pitch;

    struct
     {
       void * dst;
       size_t dpitch;
       const void * src;
       size_t spitch;
       size_t width;
       size_t height;
       enum cudaMemcpyKind kind;
     } cuda_memcpy2_d;

     struct
     {
       const void * symbol;
       const void * src;
       size_t count;
       size_t offset;
       enum cudaMemcpyKind kind;
     } cuda_memcpy_to_symbol;

     struct
     {
       int x;
       int y;
       int z;
       int w;
       enum cudaChannelFormatKind f;
       // return value non-trivial (i.e. other than cudaError_t)
       cudaChannelFormatDesc ret_val;
     } cuda_create_channel_desc;

     struct
     {
       // (*offset) is an "out" parameter. The value changes in the proxy.
       // we chose "offset" instead of "*offset", it's easier to code this way.
       size_t offset;
       struct textureReference *texref;
       const void  *devPtr;
       // we chose "desc" instead of "*desc", it's easier to code this way.
       struct cudaChannelFormatDesc desc;
       size_t width;
       size_t height;
       size_t pitch;
     } cuda_bind_texture2_d;

     /* Added to support HPGMG-CUDA */
     struct
     {
       size_t offset; // (*offset)
       size_t *offsetp; // to be used on restart.
       const textureReference *texref;
       const void *devPtr;
       struct cudaChannelFormatDesc desc; // (*desc)
       size_t size;
     } cuda_bind_texture;

    struct
    {
      /* (* pTexObject) is an "out" parameter. */
      cudaTextureObject_t pTexObject;
      cudaTextureObject_t *pTexObjectp; // useful on restart.
      /* (* pResDesc) will be passed as pointer in the proxy*/
      struct cudaResourceDesc pResDesc;
      /* (* pTexDesc) will be passed as pointer in the proxy*/
      struct cudaTextureDesc pTexDesc;
      /* (*pResViewDesc) will be passed as pointer in the proxy*/
      struct cudaResourceViewDesc pResViewDesc;
    } cuda_create_texture_object;

    struct
    {
      cudaTextureObject_t texObject;
    }cuda_destroy_texture_object;

    struct
    {
      cudaStream_t stream;
    } cuda_stream_synchronize;

    struct
    {
      cudaEvent_t event;
    } cuda_event_destroy;

    struct
    {
      const textureReference* texref;
    } cuda_unbind_texture;

    struct
    {
      cudaEvent_t event;
    } cuda_event_query;

    struct
    {
      void *ptr;
    } cuda_free_host;

    struct
    {
      int canAccessPeer; // int instead of (int *) \
                            since the value changes in the proxy.
      int device;
      int peerDevice;
    } cuda_device_can_access_peer;

    struct
    {
      int value; // int instead of (int *) since the value changes in the proxy.
      cudaDeviceAttr attr;
      int device;
    }cuda_device_get_attribute;

    struct
    {
      cudaFuncCache cacheConfig;
    } cuda_deviceSetCacheConfig;

    struct
    {
      cudaSharedMemConfig config;
    } cuda_deviceSetSharedMemConfig;

    struct
    {
      cudaEvent_t event; // event instead of (event *) since
                         // the value changes in the proxy.
      unsigned int flags;
    } cuda_eventCreateWithFlags;

    struct
    {
      cudaEvent_t event;
      cudaStream_t stream;
    } cuda_eventRecord;

    struct
    {
      cudaFuncAttributes attr; // attr instead of (* attr) since the value
                               // changes in the proxy.
      const void *func;
    } cuda_funcGetAttributes;

    struct
    {
      int device; // int instead of (int *) size the value changes in the proxy.
    } cuda_getDevice;

    struct
    {
      int count; // int instead of (int *) size the value changes in the proxy.
    } cuda_getDeviceCount;

    struct
    {
      cudaDeviceProp prop; // the value changes in the proxy.
      int device;
    } cuda_getDeviceProperties;

    struct
    {
      void *dst;
      const void *src;
      size_t count;
      cudaMemcpyKind kind;
      cudaStream_t stream;
    } cuda_memcpyAsync;

    struct
    {
      void *devPtr;
      int value;
      size_t count;
    } cuda_memset;

    struct
    {
      int  device;
    } cuda_setDevice;

    struct
    {
      void *ptr;
      cudaPointerAttributes attributes;
    } cuda_pointer_get_attributes;

    struct
    {
      int numBlocks;
      const void* func;
      int  blockSize;
      size_t dynamicSMemSize;
    } cuda_occupancy_max_active_blocks_per_multiprocessor;
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
void copy_data_to_host(void);
void copy_data_to_device(void);

#ifndef PYTHON_AUTO_GENERATE
void send_recv(int fd, cudaSyscallStructure *strce_to_send,
              cudaSyscallStructure *rcvd_strce, cudaError_t *ret_val);
void log_append(cudaSyscallStructure record);
bool log_read(cudaSyscallStructure *record);
#else

struct CudaCallLog_t {
  void *fncargs;
  size_t size;
  void *results;
  size_t resSize;
  void *host_addr;   // Valid only for a cudaMalloc region
};

EXTERNC ssize_t readAll(int fd, void *buf, size_t count);
EXTERNC ssize_t writeAll(int fd, const void *buf, size_t count);

// Just for testing purpose. Otherwise this function is obsolete.
void send_recv(int fd, cudaSyscallStructure *strce_to_send,
              cudaSyscallStructure *rcvd_strce, cudaError_t *ret_val);
void logs_read_and_apply(void (*apply)(CudaCallLog_t *l));
void log_append(void *ptr, size_t size,
                void *results, size_t resSize);
#endif

struct ShadowRegion {
  void *addr;
  size_t len;
  bool dirty;
  int prot;
};

#ifdef STATS
extern uint64_t totalTimeInSendingUVMData;
extern uint64_t totalTimeInSearchingShadowPages;
extern uint64_t totalTimeInRecvingUVMData;
#endif // ifdef STATS

ShadowRegion* getShadowRegionForAddr(void *addr);
void remove_shadow_region(void *addr);

void disable_shadow_page_flushing();
void enable_shadow_page_flushing();

void disable_cuda_call_logging();
void enable_cuda_call_logging();
bool should_log_cuda_calls();

void userfaultfd_initialize(void);
void segvfault_initialize(void);
void reset_uffd(void);
void* create_shadow_pages(size_t size, void *remoteAddr = NULL);
void unregister_all_pages();
void register_all_pages();
void protect_all_pages();
void flushDirtyPages();

// This is now static, scope within one file
// dmtcp::map<void*, void*>& shadowPageMap()fdef PYTHON_AUTO_GENERATE
//  askdaj aslkdas;
//  #endif
//  ;

#endif // ifndef STANDALONE

#endif // ifndef  __CUDA_PLUGIN_H
