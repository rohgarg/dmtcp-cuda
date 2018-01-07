// The previous design (not using auto-generated code) requireed a
//   cudaSyscallStructure *record
// where the struct would be a union of all possible structs for any CUDA call.
// In the new design, we simply use a static buffer of sufficient size.

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
