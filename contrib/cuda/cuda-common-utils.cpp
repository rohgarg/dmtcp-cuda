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

// DMTCP utils
#include "constants.h"
#include "processinfo.h"

#include "cuda_plugin.h"

int initialized = False;

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

#ifdef USE_CMA
pid_t cpid = 0;
#endif // ifdef USE_CMA

// initialize the proxy
void
proxy_initialize(void)
{
  // char sktname[10] = {0};
  // snprintf(sktname, 10, "%s%d", SKTNAME, rand()%9);
  const char *sktname = SKTNAME; // tmpnam(NULL)
  JNOTE("using socket")(sktname);
  memset(&sa_proxy, 0, sizeof(sa_proxy));
  strcpy(sa_proxy.sun_path, sktname);
  sa_proxy.sun_family = AF_UNIX;
  char *const args[] = {const_cast<char*>("../../bin/dmtcp_nocheckpoint"),
                        const_cast<char*>(dmtcp::ProcessInfo::instance()
                               .procSelfExe().c_str()),
                        const_cast<char*>(sktname),
                        NULL}; // FIXME: Compiler warning

  setenv("CUDA_PROXY_SOCKET", sktname, 1);
  switch (_real_fork()) {
    case -1:
      JASSERT(false)(JASSERT_ERRNO).Text("Failed to fork cudaproxy");

    case 0:
      setenv(ENV_VAR_ORIG_LD_PRELOAD, "./libcudaproxy.so", 1);
      setenv("CUDA_PROXY_SOCKET", sktname, 1);
      JASSERT(execvp((const char*)args[0], args) != -1)(JASSERT_ERRNO)
             .Text("Failed to exec cudaproxy");
  }

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

#ifdef USE_CMA
  JASSERT(readAll(skt_master, &cpid, sizeof(cpid)) == sizeof(cpid) && cpid > 0);
#endif // ifdef USE_CMA

  initialized = True;
}

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
  JASSERT(num_written == count) (num_written) (count);
  return num_written;
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
