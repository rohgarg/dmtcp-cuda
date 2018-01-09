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
  unregister_all_pages();
  copy_data_to_host();
}

static void
resume()
{
  register_all_pages();
}


static void
restart()
{
  JTRACE("Trying to re-init the CUDA driver");
  close(skt_master);
  proxy_initialize();
  reset_uffd();
  register_all_pages();
  logFd = open(LOGFILE, O_APPEND|O_RDWR);
  if (logFd == -1)
  {
    perror("open()");
    exit(EXIT_FAILURE);
  }
  disable_cuda_call_logging();

  // Replay calls from the log
  void *log_read_buf;
  size_t buf_size;
  enum cuda_op op;
  log_read_buf = log_read(&buf_size);
  memcpy(&op, log_read_buf, sizeof(op));

  while (op != OP_LAST_FNC) {
    // TODO: Add cases for other calls that should be replayed.
    if (op == OP_cudaMalloc) {
      // read size. "size" as in cudaMalloc(void **ptr, "size")
      size_t size;
      memcpy(&size, log_read_buf+sizeof(op), sizeof(size));
      /*
      The assumption is that when cudaMalloc is replayed the same
      virtual address will be returned. If that is the case then
      the application can safely use the same virtual address
      on restart so long as the call has been replayed.
      */

      /*
      "ptr" is not used anywhere, it is solely for reply purpose.
      */
      void *ptr;
      cudaMalloc(&ptr, size);
    }

    log_read_buf = log_read(&buf_size);
    memcpy(&op, log_read_buf, sizeof(op));
  }
  copy_data_to_device();
  enable_cuda_call_logging();
}

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
