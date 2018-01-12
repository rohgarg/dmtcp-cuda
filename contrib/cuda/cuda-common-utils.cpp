#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>

#include <sys/stat.h>
#include <fcntl.h>

#if USE_SHM
# include <sys/ipc.h>
# include <sys/shm.h>
#endif

// DMTCP utils
#include "constants.h"
#include "processinfo.h"

#include "cuda_plugin.h"

int initialized = False;
#if USE_SHM
void *shmaddr = NULL;
#endif

// for pinned memory
// #define PINNED_MEM_MAX_ALLOC 100
typedef struct pseudoPinnedMem
{
  void *array[PINNED_MEM_MAX_ALLOC];
  int index;
} pseudoPinnedMem_t;
pseudoPinnedMem_t pseudoPinnedMemArray = {array : {}, index : 0};

// master socket
int skt_master;

// proxy address
struct sockaddr_un sa_proxy;

// NOTE: Do not access this directly; use the accessor functions
bool enableCudaCallLogging = true;

// initialize the proxy
void proxy_initialize(void)
{
  memset(&sa_proxy, 0, sizeof(sa_proxy));
  strcpy(sa_proxy.sun_path, SKTNAME);
  sa_proxy.sun_family = AF_UNIX;
  char *const args[] = {const_cast<char*>("../../bin/dmtcp_nocheckpoint"),
                        const_cast<char*>(dmtcp::ProcessInfo::instance()
                               .procSelfExe().c_str()),
                        const_cast<char*>(SKTNAME),
                        NULL}; // FIXME: Compiler warning

  switch (_real_fork()) {
    case -1:
      JASSERT(false)(JASSERT_ERRNO).Text("Failed to fork cudaproxy");

    case 0:
      setenv(ENV_VAR_ORIG_LD_PRELOAD, "./libcudaproxy.so", 1);
      setenv("CUDA_PROXY_SOCKET", "./proxy", 1);
      JASSERT(execvp((const char*)args[0], args) != -1)(JASSERT_ERRNO)
             .Text("Failed to exec cudaproxy");
  }

#if USE_SHM
  // create shared memory
  key_t shmKey;
  JASSERT((shmKey = ftok(".", 1) != -1)(JASSERT_ERRNO);

  int shm_flags = IPC_CREAT | 0666;
  int shmID;
  JASSERT((shmID = shmget(shmKey, SHMSIZE, shm_flags)) != -1)(JASSERT_ERRNO);

  JASSERT((shmaddr = shmat(shmID, NULL, 0)) != (void *)-1)(JASSERT_ERRNO);
#endif

  // connect to the proxy:server
  JASSERT((skt_master = socket(AF_UNIX, SOCK_STREAM, 0)) > 0)(JASSERT_ERRNO);

  while (connect(skt_master, (struct sockaddr *)&sa_proxy, sizeof(sa_proxy))
         == -1) {
    if (errno = ENOENT) {
      sleep(1);
      continue;
    } else {
      JASSERT(false)(JASSERT_ERRNO).Text("Failed to connect with proxy");
    }
  }

#if USE_SHM
  // Send the shmID to the proxy
  int realId = dmtcp_virtual_to_real_shmid(shmID);
  JASSERT(write(skt_master, &realId, sizeof(shmID)) != -1)(JASSERT_ERRNO);
#endif

  initialized = True;
}

#ifndef PYTHON_AUTO_GENERATE
    // Old non-auto-generated version
// open the log file and
// append a cuda system call structure to it
void log_append(cudaSyscallStructure record)
{
  printf("**** OBSOLETE:  Please re-write to use log/replay style from"
         "log_append_and_read.cpp\n");
  JASSERT(write(logFd, &record, sizeof(record)) != -1)(JASSERT_ERRNO);
}

bool log_read(cudaSyscallStructure *record)
{
  printf("**** OBSOLETE:  Please re-write to use log/replay style from"
         "log_append_and_read.cpp\n");
  int ret = read(logFd, record, sizeof(*record));
  if (ret == -1) {
    JASSERT(false)(JASSERT_ERRNO);
  }
  if (ret == 0 || ret < sizeof(*record)) {
    return false;
  }
  return true;
}
#else
// The previous design (not using auto-generated code) required
//   cudaSyscallStructure *record
// where the struct would be a union of all possible structs for any CUDA call.
// In the new design, we simply use a static buffer of sufficient size,
//   LOG_READ_BUF_SIZE.

// open the log file and
// append a cuda system call structure to it
// RECOMMENDED USAGE FOR WRITING LAST LOG VALUE BEFORE CKPT:
//    enum cuda_op tmp = OP_LAST_FNC;
//    log_append(&tmp, sizeof tmp);
void log_append(void *ptr, size_t size) {
  JASSERT(write(logFd, &size, sizeof(size)) != sizeof(size))(JASSERT_ERRNO);
  JASSERT(write(logFd, ptr, size) != size)(JASSERT_ERRNO);
}

#define LOG_READ_BUF_SIZE 1000
static int log_read_buf[LOG_READ_BUF_SIZE]; // Assumes that 1000 bytes suffices
                                     // as long as we don't replay cudaMalloc.

// NOTE: Caller must use result buffer, before calling log_read() again.
// RECOMMENDED USAGE FOR REPLAY TO PROXY:
//    size_t size;
//    void *buf = log_read(&size);
//    if ((*(enum cuda_op *)buf) == OP_LAST_FNC) {
//      return; // Nothing more exists in the log.
//    }
//    write(skt_master, buf, size);  // Replay it in the proxy.
void * log_read(size_t *size) { // size is an "out" parameter
  JASSERT(size != NULL);
  JASSERT(read(logFd, size, sizeof(*size)) != sizeof(*size))(JASSERT_ERRNO);
  JASSERT(*size <= LOG_READ_BUF_SIZE);
  JASSERT(read(logFd, log_read_buf, *size) != *size)(JASSERT_ERRNO);
  return log_read_buf;
}
#endif

/*
  This function sends to the proxy the structure with cuda syscall parameters.
  It then receives the return value and gets the structure back.
*/
void send_recv(int fd, cudaSyscallStructure *strce_to_send,
               cudaSyscallStructure *rcvd_strce, cudaError_t *ret_val)
{
  // send the structure
  JASSERT(write(fd, strce_to_send, sizeof(cudaSyscallStructure)) != -1)
         (JASSERT_ERRNO);

  if (strce_to_send->payload) {
    JASSERT(write(fd, strce_to_send->payload, strce_to_send->payload_size) !=
            -1)(strce_to_send->payload)(JASSERT_ERRNO);
  }

  // receive the result
  JASSERT(read(fd, ret_val, sizeof(int)) != -1)(JASSERT_ERRNO);


  if (strce_to_send->op != CudaGetLastError){
    JASSERT((*(cudaError_t*)ret_val) == cudaSuccess)
       (cudaGetErrorString(*(cudaError_t*)ret_val)).Text("CUDA syscall failed");
  }

  // get the structure back
  memset(rcvd_strce, 0, sizeof(cudaSyscallStructure));
  JASSERT(read(fd, rcvd_strce, sizeof(cudaSyscallStructure)) != -1)
         (JASSERT_ERRNO);

  switch(rcvd_strce->op)
  {
    case CudaGetErrorString:
    {
      // "cudaGetErrorString" is a special case, its return type
      // is "const char *".
      size_t size = (rcvd_strce->syscall_type).cuda_get_error_string.size;
      char *error_string = (char *) malloc(size * sizeof(char));
      JASSERT(read(fd, error_string, size) != -1) (JASSERT_ERRNO);
      (rcvd_strce->syscall_type).cuda_get_error_string.error_string \
       = error_string;
    }
    break;
  }
}

void
disable_cuda_call_logging()
{
  // TODO: Add locks for thread safety
  enableCudaCallLogging = false;
}

void
enable_cuda_call_logging()
{
  // TODO: Add locks for thread safety
  enableCudaCallLogging = true;
}

bool
should_log_cuda_calls()
{
  // TODO: Add locks for thread safety
  return enableCudaCallLogging;
}

// For pinned Memory
void
pseudoPinnedMem_append(void *ptr) {
  pseudoPinnedMemArray.array[pseudoPinnedMemArray.index++] = ptr;

  return;
}

bool
is_pseudoPinnedMem(void *ptr) {
  int index = pseudoPinnedMemArray.index;
  for (int i=0; i <= index; ++i){
    if (pseudoPinnedMemArray.array[pseudoPinnedMemArray.index] == ptr)
      return true;
  }

  return false;
}

void
pseudoPinnedMem_remove(void *ptr) {
  pseudoPinnedMemArray.array[pseudoPinnedMemArray.index--] = ptr;

  return;
}
