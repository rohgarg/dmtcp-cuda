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

#include "cuda_plugin.h"

int initialized = False;
#if USE_SHM
void *shmaddr = NULL;
#endif

// master socket
int skt_master;

// proxy address
struct sockaddr_un sa_proxy;

// initialize the proxy
void proxy_initialize(void)
{
  memset(&sa_proxy, 0, sizeof(sa_proxy));
  strcpy(sa_proxy.sun_path, SKTNAME);
  sa_proxy.sun_family = AF_UNIX;
  char *const args[] = {const_cast<char*>("../../bin/dmtcp_nocheckpoint"),
                        const_cast<char*>("./cudaproxy"),
                        const_cast<char*>(SKTNAME),
                        NULL}; // FIXME: Compiler warning

  switch (_real_fork()) {
    case -1:
      JASSERT(false)(JASSERT_ERRNO).Text("Failed to fork cudaproxy");

    case 0:
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

// open the log file and
// append a cuda system call structure to it
void log_append(cudaSyscallStructure record)
{
  JASSERT(write(logFd, &record, sizeof(record)) != -1)(JASSERT_ERRNO);
}

bool log_read(cudaSyscallStructure *record)
{
  int ret = read(logFd, record, sizeof(*record));
  if (ret == -1) {
    JASSERT(false)(JASSERT_ERRNO);
  }
  if (ret == 0 || ret < sizeof(*record)) {
    return false;
  }
  return true;
}

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
    // FIXME: Ugly hack to detect if it's a pointer on a 64-bit system
    if (strce_to_send->payload_size == 8) {
      void *origPtr = (void*)*(unsigned long*)strce_to_send->payload;
      void *dataPage = getAlignedAddress((uintptr_t)origPtr,
                                         page_size > 0 ? page_size : 4096);
      void *payload = NULL;
      if (shadowPageMap().find(dataPage) != shadowPageMap().end()) {
        payload = (void*)((uintptr_t)shadowPageMap()[dataPage] +
                          ((uintptr_t)origPtr -(uintptr_t)dataPage));
      }
      JASSERT(write(fd, &payload, sizeof(payload)) != -1)(JASSERT_ERRNO);
    } else {
      JASSERT(write(fd, strce_to_send->payload, strce_to_send->payload_size) !=
              -1)(JASSERT_ERRNO);
    }
  }

  // receive the result
  JASSERT(read(fd, ret_val, sizeof(int)) != -1)(JASSERT_ERRNO);

  JASSERT((*ret_val) == cudaSuccess).Text("CUDA syscall failed");

  // get the structure back
  memset(rcvd_strce, 0, sizeof(cudaSyscallStructure));
  JASSERT(read(fd, rcvd_strce, sizeof(cudaSyscallStructure)) != -1)
         (JASSERT_ERRNO);
}
