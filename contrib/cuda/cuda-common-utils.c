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
#if 0 // Disable this for now. Manually launch the proxy from another terminal
  char *const args[] = {"../..//dmtcp_nocheckpoint",
                        "./cudaproxy",
                        SKTNAME,
                        NULL};

  switch (_real_fork())
  {
    case -1:
      perror("fork()");
      exit(EXIT_FAILURE);

    case 0:
      if (execvp((const char*)args[0], args) == -1)
      {
        perror("execvp");
        exit(EXIT_FAILURE);
      }
  }
#endif
  
#if USE_SHM
  // create shared memory
  key_t shmKey;
  if ((shmKey = ftok(".", 1) == -1))
  {
    perror("ftok()");
    exit(EXIT_FAILURE);
  }

  int shm_flags = IPC_CREAT | 0666;
  int shmID;
  if ((shmID = shmget(shmKey, SHMSIZE, shm_flags)) == -1)
  {
    perror("shmget()");
    exit(EXIT_FAILURE);
  }
  
  if ((shmaddr = shmat(shmID, NULL, 0)) == (void *)-1)
  {
    perror("shmat()");
    exit(EXIT_FAILURE);
  }
#endif

  // connect to the proxy:server
   if ((skt_master = socket(AF_UNIX, SOCK_STREAM, 0)) == 1)
  {
    perror("socket()");
    exit(EXIT_FAILURE);
  }

  while (connect(skt_master, (struct sockaddr *)&sa_proxy, sizeof(sa_proxy)) == -1)
  {
    if (errno = ENOENT)
    {
      sleep(1);
      continue;
    }

    else
      exit(EXIT_FAILURE);
  }
  
#if USE_SHM
  // Send the shmID to the proxy
  int realId = dmtcp_virtual_to_real_shmid(shmID);
  if (write(skt_master, &realId, sizeof(shmID)) == -1)
  {
    perror("write()");
    exit(EXIT_FAILURE);
  }
#endif

  initialized = True;
}

// open the log file and
// append a cuda system call structure to it
void log_append(cudaSyscallStructure record)
{
  if (write(logFd, &record, sizeof(record)) == -1)
  {
    perror("write()");
    exit(EXIT_FAILURE);
  }
}

int log_read(cudaSyscallStructure *record)
{
  int ret = read(logFd, record, sizeof(*record));
  if (ret == -1)
  {
    perror("write()");
    exit(EXIT_FAILURE);
  }
  if (ret == 0 || ret < sizeof(*record)) {
    return False;
  }
  return True;
}

/*
  This function sends to the proxy the structure with cuda syscall parameters.
  It then receives the return value and gets the structure back.
*/
void send_recv(int fd, cudaSyscallStructure *strce_to_send,
              cudaSyscallStructure *rcvd_strce, cudaError_t *ret_val)
{
   // send the structure
  if (write(fd, strce_to_send, sizeof(cudaSyscallStructure)) == -1)
  {
    perror("write()");
    exit(EXIT_FAILURE);
  }

  if (strce_to_send->payload) {
    if (write(fd, strce_to_send->payload, strce_to_send->payload_size) == -1) {
      perror("write()");
      exit(EXIT_FAILURE);
    }
  }

   // receive the result
  if (read(fd, ret_val, sizeof(int)) == -1)
  {
    perror("read()");
    exit(EXIT_FAILURE);
  }

  if ((*ret_val) != cudaSuccess)
  {
    printf("cuda syscall failed\n");
    exit(EXIT_FAILURE);
  }

  // get the structure back
  memset(rcvd_strce, 0, sizeof(cudaSyscallStructure));
  if (read(fd, rcvd_strce, sizeof(cudaSyscallStructure)) == -1)
  {
    perror("read()");
    exit(EXIT_FAILURE);
  }
}
