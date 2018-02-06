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
    // logFd = open(LOGFILE, O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
    // if (logFd == -1)
    // {
    //   perror("open()");
    //   exit(EXIT_FAILURE);
    // }
    // close(logFd);
    // logFd = open(LOGFILE, O_APPEND|O_RDWR);
    // if (logFd == -1)
    // {
    //   perror("open()");
    //   exit(EXIT_FAILURE);
    // }
    break;
  }
  case DMTCP_EVENT_EXIT:
  {
    JTRACE("The plugin is being called before exiting.");
#ifdef STATS
    JNOTE("Runtime stats")(totalTimeInRecvingUVMData)(totalTimeInSendingUVMData)
         (totalTimeInSearchingShadowPages);
    print_stats();
#endif // ifdef STATS
#ifdef USE_SHM
    enum cuda_op op = OP_LAST_FNC;
    memcpy(shared_mem_ptr, &op, sizeof op);
    unlock_proxy();
#endif // ifdef USE_SHM
    break;
  }
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
  disable_shadow_page_flushing();
  copy_data_to_host();
  // enum cuda_op tmp = OP_LAST_FNC;
  // log_append(&tmp, sizeof tmp);
}

static void
resume()
{
  register_all_pages();
  enable_shadow_page_flushing();
}

#define LOG_READ_BUF_SIZE 1000

static void
restart()
{
  JTRACE("Trying to re-init the CUDA driver");
  close(skt_master);
  proxy_initialize();
#ifdef USERFAULTFD
  reset_uffd();
#endif
  // logFd = open(LOGFILE, O_APPEND|O_RDWR);
  // if (logFd == -1)
  // {
  //   perror("open()");
  //   exit(EXIT_FAILURE);
  // }
  disable_cuda_call_logging();
  disable_shadow_page_flushing();

  // Replay calls from the log
  copy_data_to_device();
  // After flushing the data to the device, we restore
  // the perms on the shadow pages to what they were
  // pre-checkpoint
  register_all_pages();
  enable_cuda_call_logging();
  enable_shadow_page_flushing();
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
