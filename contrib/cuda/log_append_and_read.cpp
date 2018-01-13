#include "cuda_plugin.h"

// struct CudaCallLog_t {
//   void *fncargs;
//   size_t size;
//   void *results;
//   size_t resSize;
//   void *host_addr;   // Valid only for a cudaMalloc region
// };

static dmtcp::vector<CudaCallLog_t>&
cudaCallsLog()
{
  static dmtcp::vector<CudaCallLog_t> *instance = NULL;
  if (instance == NULL) {
    void *buffer = JALLOC_MALLOC(1024 * 1024);
    instance = new (buffer)dmtcp::vector<CudaCallLog_t>();
  }
  return *instance;
}

// The previous design (not using auto-generated code) requireed a
//   cudaSyscallStructure *record
// where the struct would be a union of all possible structs for any CUDA call.
// In the new design, we simply use a static buffer of sufficient size.

// open the log file and
// append a cuda system call structure to it
// RECOMMENDED USAGE FOR WRITING LAST LOG VALUE BEFORE CKPT:
//    enum cuda_op tmp = OP_LAST_FNC;
//    log_append(&tmp, sizeof tmp);
// This function does in-memory logging of CUDA calls that are specified
// using the CUDA_WRAPPER_WITH_LOGGING tag.
void log_append(void *ptr, size_t size,
                void *results = NULL, size_t resSize = 0) {
  if (should_log_cuda_calls()) {
    CudaCallLog_t log = {0};
    if (size > 0) {
      void *args = JALLOC_HELPER_MALLOC(size);
      memcpy(args, ptr, size);
      log.size = size;
      log.fncargs = args;
    }
    if (resSize > 0) {
      void *res = JALLOC_HELPER_MALLOC(resSize);
      memcpy(res, results, resSize);
      log.resSize = resSize;
      log.results = res;
    }
    cudaCallsLog().push_back(log);
  }
}

// This function iterates over the CUDA calls log and calls the given
// function on each call log object
void logs_read_and_apply(void (*apply)(CudaCallLog_t *l))
{
  dmtcp::vector<CudaCallLog_t>::iterator it;
  for (it = cudaCallsLog().begin(); it != cudaCallsLog().end(); it++) {
    apply(&(*it));
  }
}
