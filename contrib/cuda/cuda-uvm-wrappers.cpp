#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include "config.h"
#include "cuda_plugin.h"

// XXX: cudaMallocManaged is more or less the same as cudaMalloc, except
// for the additional creation of shadow pages.
EXTERNC cudaError_t
cudaMallocManaged(void **pointer, size_t size, unsigned int flags)
{
  if (!initialized)
    proxy_initialize();

#ifdef USERFAULTFD_INITIALIZED
  if (!ufd_initialized)
    userfaultfd_initialize();
#else
  if (!segvfault_initialized)
    segvfault_initialize();
#endif

  cudaError_t ret_val;
  char send_buf[1000];
  char recv_buf[1000];
  int chars_sent = 0;
  int chars_rcvd = 0;

  // Write the IN arguments to the proxy
  enum cuda_op op = OP_cudaMallocManaged;
  memcpy(send_buf + chars_sent, &op, sizeof op);
  chars_sent += sizeof(enum cuda_op);
  memcpy(send_buf + chars_sent, & size, sizeof size);
  chars_sent += sizeof size;
  memcpy(send_buf + chars_sent, & flags, sizeof flags);
  chars_sent += sizeof flags;

  // Send op code and args to proxy
  JASSERT(write(skt_master, send_buf, chars_sent) == chars_sent)
         (JASSERT_ERRNO);

  // Receive the OUT arguments after the proxy made the function call
  // Compute total chars_rcvd to be read in the next msg
  chars_rcvd = sizeof *pointer;
  chars_rcvd += sizeof ret_val;
  JASSERT(read(skt_master, recv_buf, chars_rcvd) == chars_rcvd)
         (JASSERT_ERRNO);

  // Extract OUT variables
  chars_rcvd = 0;
  memcpy(pointer, recv_buf + chars_rcvd, sizeof *pointer);
  chars_rcvd += sizeof *pointer;

  memcpy(&ret_val, recv_buf + chars_rcvd, sizeof ret_val);
  JASSERT(ret_val == cudaSuccess)\
         (ret_val).Text("Failed to create UVM region");

  // change the pointer to point the global memory (device memory)
  *pointer = create_shadow_pages(size, *pointer);

  // TODO:  Add logging

  return ret_val;
}
