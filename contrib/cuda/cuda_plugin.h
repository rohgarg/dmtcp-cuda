#ifndef __CUDA_PLUGIN_H
#define __CUDA_PLUGIN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <sys/time.h>

#include "dmtcp.h"

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
EXTERNC ssize_t readAll(int fd, void *buf, size_t count);
EXTERNC ssize_t writeAll(int fd, const void *buf, size_t count);

#ifndef STANDALONE

// The following declarations, definitions, etc. are only
// valid for the DMTCP CUDA plugin

#include <stdint.h>

#include "jassert.h"
#include "dmtcp_dlsym.h"

#define   _real_fork      NEXT_FNC_DEFAULT(fork)

#define True    1
#define False   0

#define SKTNAME "proxy"
#define LOGFILE "cudaSysCallsLog"

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

static inline void*
getAlignedAddress(uintptr_t ptr, size_t alignment)
{
  const size_t mask = alignment - 1;
  return (void *) (ptr & ~mask);
}

void proxy_initialize(void);
void copy_data_to_host(void);
void copy_data_to_device(void);

struct CudaCallLog_t {
  void *fncargs;
  size_t size;
  void *results;
  size_t resSize;
  void *host_addr;   // Valid only for a cudaMalloc region
};

// Just for testing purpose. Otherwise this function is obsolete.
void logs_read_and_apply(void (*apply)(CudaCallLog_t *l));
void log_append(void *ptr, size_t size,
                void *results, size_t resSize);

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

#ifdef USE_CMA
// DMTCP CUDA plugin uses this to store the pid of the
// forked child proxy process. This is then used for
// doing CMA transfers of UVM data.
extern pid_t cpid;
#endif // ifdef USE_CMA

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

#endif // ifndef STANDALONE

#endif // ifndef  __CUDA_PLUGIN_H
