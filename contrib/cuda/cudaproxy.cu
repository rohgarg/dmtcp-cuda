#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include <cuda_profiler_api.h>

// Definitions of common structs shared with the main process
#include "cuda_plugin.h"
#include "trampolines.h"

#define SKTNAME "proxy"
#define specialCudaReturnValue  60000

size_t totalRead = 0;
size_t totalWritten = 0;

static pid_t ppid = 0;

#ifdef USE_SHM
// for mutex, cond. var., etc
int shmid;
char *shmptr;

// for actual data
int shared_mem_id;
char *shared_mem_ptr;

pthread_cond_t *cvptr;
pthread_condattr_t cattr;
pthread_mutex_t    *mptr;
pthread_mutexattr_t matr;

enum turn *current_turn;
#endif // ifdef USE_SHM

#ifdef STATS
struct CallCost costs[OP_LAST_FNC] = {0};
uint64_t totalTimeInUvmCopy = 0;
std::map<uint64_t, uint64_t> uvmReadReqMap;
std::map<uint64_t, uint64_t> uvmWriteReqMap;
#endif // ifdef STATS

EXTERNC ssize_t
readAll(int fd, void *buf, size_t count)
{
  ssize_t rc;
  char *ptr = (char *)buf;
  size_t num_read = 0;

  for (num_read = 0; num_read < count;) {
    rc = read(fd, ptr + num_read, count - num_read);
    if (rc == -1) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      } else {
        return -1;
      }
    } else if (rc == 0) {
      break;
    } else { // else rc > 0
      num_read += rc;
    }
  }
  totalRead += num_read;
  return num_read;
}

EXTERNC ssize_t
writeAll(int fd, const void *buf, size_t count)
{
  const char *ptr = (const char *)buf;
  size_t num_written = 0;

  do {
    ssize_t rc = write(fd, ptr + num_written, count - num_written);
    if (rc == -1) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      } else {
        return rc;
      }
    } else if (rc == 0) {
      break;
    } else { // else rc > 0
      num_written += rc;
    }
  } while (num_written < count);
  totalWritten += num_written;
  return num_written;
}


void
print_stats()
{
  printf("Total: Sent: %zu; Received: %zu\n", totalRead, totalWritten);
#ifdef STATS
  printf("Total: Total Time In Uvm Copy: %zu\n", totalTimeInUvmCopy);
  printf("Total: Write request map:\n");
  for(std::map<uint64_t, uint64_t>::const_iterator it = uvmWriteReqMap.begin();
      it != uvmWriteReqMap.end(); ++it) {
    printf("%d, %d\n", it->first, it->second);
  }
  printf("Total: Read request map:\n");
  for(std::map<uint64_t, uint64_t>::const_iterator it = uvmReadReqMap.begin();
      it != uvmReadReqMap.end(); ++it) {
    printf("%d, %d\n", it->first, it->second);
  }
  printf("Total: CUDA call cost:\n");
  for (int i = 0; i < OP_LAST_FNC; i++) {
    if (costs[i].count > 0) {
       printf("%d, %d, %llu, %llu\n", i, costs[i].count, costs[i].totalTime, costs[i].cudaCallCost);
    }
  }
#endif
}

# include "python-auto-generate/cudaproxy.icu"

static trampoline_info_t main_trampoline_info;

int skt_accept;
static int start_proxy(void);


// This is the trampoline destination for the user main; this does not return
// to the user main function.
int main_wrapper()
{
  start_proxy();
  return 0;
}

__attribute__((constructor))
void proxy_init()
{
  void *handle = dlopen(NULL, RTLD_NOW);
  void *addr = dlsym(handle, "main");
  assert(addr != NULL);
  dmtcp_setup_trampoline_by_addr(addr, (void*)&main_wrapper, &main_trampoline_info);
}

static int start_proxy(void)
{
  // set up the server
  int skt_proxy;
  struct sockaddr_un sa_proxy;
#ifdef USE_CMA
  pid_t pid = getpid();
#endif // ifdef USE_CMA
  const char *sktname = getenv("CUDA_PROXY_SOCKET");
  if (!sktname) {
    sktname = SKTNAME;
  }

  (void) unlink(sktname);
  memset(&sa_proxy, 0, sizeof(sa_proxy));
  strcpy(sa_proxy.sun_path, sktname);
  sa_proxy.sun_family = AF_UNIX;

  if ((skt_proxy = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
  {
    perror("socket()");
    exit(EXIT_FAILURE);
  }

  if (bind(skt_proxy, (struct sockaddr *)&sa_proxy, sizeof(sa_proxy)) == -1)
  {
    perror("bind()");
    exit(EXIT_FAILURE);
  }

  if (listen(skt_proxy, SOMAXCONN) == -1)
  {
    perror("listen()");
    exit(EXIT_FAILURE);
  }
 
  if ((skt_accept = accept(skt_proxy, NULL, 0)) == -1)
  {
    perror("accept()");
    exit(EXIT_FAILURE);
  }

  ppid = getppid();
  assert(ppid != -1);

#ifdef USE_SHM
  if ((shmid = shmget(IPC_PRIVATE, SHM_REGION_1_SIZE,
                      IPC_EXCL | IPC_CREAT | 0660)) < 0)
    perror("shmget"), exit(1) ;

  if ((shmptr = (char *)shmat(shmid, (void *)0, 0)) == NULL)
    perror("shmat"), exit(1);

  if ((shared_mem_id = shmget(IPC_PRIVATE, SHM_REGION_2_SIZE,
                              IPC_EXCL | IPC_CREAT | 0660)) < 0)
    perror("shmget"), exit(1) ;

  if ((shared_mem_ptr = (char *)shmat(shared_mem_id, (void *)0, 0)) == NULL)
    perror("shmat"), exit(1);

  cvptr = (pthread_cond_t *)shmptr;
  mptr = (pthread_mutex_t *)((char*)cvptr + sizeof(*cvptr));
  current_turn = (enum turn*)((char*)mptr + sizeof(*mptr));

  pthread_mutexattr_init(&matr);
  pthread_mutexattr_setpshared(&matr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(mptr, &matr);

  pthread_condattr_init(&cattr);
  pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
  pthread_cond_init(cvptr, &cattr);
#endif // ifdef USE_SHM

#ifdef USE_CMA
  assert(writeAll(skt_accept, &pid, sizeof pid) == sizeof pid);
#endif // ifdef USE_CMA
#ifdef USE_SHM
  assert(writeAll(skt_accept, &shmid, sizeof shmid) == sizeof shmid);
  assert(writeAll(skt_accept, &shared_mem_id, sizeof shared_mem_id) == sizeof shared_mem_id);
#endif // ifdef USE_SHM

  // do_work() has an infinite 'while(1)' loop.
  do_work(); // never returns
  return 0;  // To satisfy compiler
}
