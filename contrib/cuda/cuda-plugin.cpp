#include <stdio.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "config.h"
#include "cuda_plugin.h"


/* Globals */
int logFd = -1;

static void
cuda_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  /* NOTE:  See warning in plugin/README about calls to printf here. */
  switch (event) {
  case DMTCP_EVENT_INIT:
  {
    JTRACE("The CUDA plugin has been initialized.");
    // create the log file
    logFd = open(LOGFILE, O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
    if (logFd == -1)
    {
      perror("open()");
      exit(EXIT_FAILURE);
    }
    close(logFd);
    logFd = open(LOGFILE, O_APPEND|O_WRONLY);
    if (logFd == -1)
    {
      perror("open()");
      exit(EXIT_FAILURE);
    }
    break;
  }
  case DMTCP_EVENT_EXIT:
    JTRACE("The plugin is being called before exiting.");
    break;
  default:
    break;
  }
}

/*
 * Global barriers
 */

static void
pre_ckpt()
{
  JTRACE("Nothing to do for now");
}

static void
resume()
{
  JTRACE("Nothing to do for now");
}


static void
restart()
{
  JTRACE("Trying to re-init the CUDA driver");
  proxy_initialize();
  logFd = open(LOGFILE, O_APPEND|O_RDWR);
  if (logFd == -1)
  {
    perror("open()");
    exit(EXIT_FAILURE);
  }
  cudaSyscallStructure rec;
  memset(&rec, 0, sizeof(rec));
  // Replay calls from the log
  bool ret = log_read(&rec);
  while (ret) {
    // TODO: Add cases for other calls
    if (rec.op == CudaMalloc) {
      cudaMalloc((void**)rec.syscall_type.cuda_malloc.pointer,
                 rec.syscall_type.cuda_malloc.size);
    }
    ret = log_read(&rec);
  }
}

/*
 * Wrapper functions
 */

#if 0
cudaError_t
cudaMalloc(void **devPtr, size_t  size)
{
  cudaError_t ret;
  ret = _real_cudaMalloc(cudaMallocPtrs[idx], cudaMallocBufSizes[idx]);
  return ret;
}

cudaError_t
cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                  size_t sharedMem, cudaStream_t stream)
{
  return _real_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t
cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
  return _real_cudaFuncGetAttributes(attr, func);
}

cudaError_t
cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
  return _real_cudaSetupArgument(arg, size, offset);
}

cudaError_t
cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                 void** args, size_t sharedMem, cudaStream_t stream)
{
  return _real_cudaLaunchKernel(func, gridDim, blockDim,
                                args, sharedMem, stream);
}

cudaError_t
cudaLaunch(const void *entry)
{
  return _real_cudaLaunch(entry);
}
#endif

static DmtcpBarrier cudaPluginBarriers[] = {
  { DMTCP_GLOBAL_BARRIER_PRE_CKPT, pre_ckpt, "checkpoint" },
  { DMTCP_GLOBAL_BARRIER_RESUME, resume, "resume" },
  { DMTCP_GLOBAL_BARRIER_RESTART, restart, "restart" }
};

DmtcpPluginDescriptor_t cuda_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "cuda",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "CUDA plugin",
  DMTCP_DECL_BARRIERS(cudaPluginBarriers),
  cuda_event_hook
};

DMTCP_DECL_PLUGIN(cuda_plugin);
