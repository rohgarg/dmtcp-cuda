#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef USERFAULTFD_DEFINED
#include <linux/userfaultfd.h>
#endif
#include <pthread.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <stdint.h>


#include "config.h"
#include "cuda_plugin.h"
#include "util.h"

// Public global vars

int page_size = -1;
int SHIFT = -1;

bool haveDirtyPages = false;

// Private global vars

struct ShadowRegion {
  void *addr;
  size_t len;
  bool dirty;
};

// Private functions

static dmtcp::map<void*, void*>&
shadowPageMap()
{
  static dmtcp::map<void*, void*> *instance = NULL;
  if (instance == NULL) {
    void *buffer = JALLOC_MALLOC(1024 * 1024);
    instance = new (buffer)dmtcp::map<void*, void*>();
  }
  return *instance;
}

static dmtcp::vector<ShadowRegion>&
allShadowRegions()
{
  static dmtcp::vector<ShadowRegion> *instance = NULL;
  if (instance == NULL) {
    void *buffer = JALLOC_MALLOC(1024 * 1024);
    instance = new (buffer)dmtcp::vector<ShadowRegion>();
  }
  return *instance;
}

/* This works for both userfaultfd-based and segfault handler-based implementations.
 *
 * In the case of the former, it adds the shadow page to the UFFD managed regions.
 *
 * In the case of the latter, it does nothing, except adding the shadow page to
 * the list of all shadow page regions. This list is used later to flush dirty pages
 * to the proxy and receiving data from the proxy.
 */
static void
monitor_pages(void *addr, size_t size, cudaSyscallStructure *remoteInfo = NULL)
{
#ifdef USERFAULTFD
  struct uffdio_register uffdio_register;

  uffdio_register.range.start = (uintptr_t)addr;
  uffdio_register.range.len = size;
  uffdio_register.mode = UFFDIO_REGISTER_MODE_MISSING;
#endif

  JNOTE("register region")(addr)(size);

#ifdef USERFAULTFD
  JASSERT(ioctl(uffd, UFFDIO_REGISTER, &uffdio_register) != -1)(JASSERT_ERRNO);
#endif

  if (remoteInfo) {
    // Save the location and size of the shadow region
    for (int i = 0; i < size / page_size; i++) {
      ShadowRegion r =  {.addr = (void*)((uintptr_t)addr + i*page_size),
                         .len = size,
                         .dirty = false};
      allShadowRegions().push_back(r);
      // Save the actual UVM region on the proxy
      shadowPageMap()[addr] = (void*)((uintptr_t)remoteInfo->
                                      syscall_type.cuda_malloc.pointer +
                                      i * page_size);
    }
  }
}

/*
 * This gets called at a synchronization point by flushDirtyPages() to send
 * to the proxy process.
 */
static bool
sendDataToProxy(void *remotePtr, void *localPtr, size_t size)
{
  cudaSyscallStructure strce_to_send, rcvd_strce;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  memset(&rcvd_strce, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocManagedMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = remotePtr;
  strce_to_send.syscall_type.cuda_memcpy.source = localPtr;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = cudaMemcpyHostToDevice;
  strce_to_send.payload = localPtr;
  strce_to_send.payload_size = size;

  // send the structure
  cudaError_t ret_val;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  JASSERT(ret_val == cudaSuccess)(ret_val)
          .Text("Failed to send UVM dirty pages");
}

/*
 * This gets called on a read fault to read in the updated data from
 * the proxy. The data is copied into the local buffer that generated
 * the segfault.
 */
static bool
receiveDataFromProxy(void *remotePtr, void *localPtr, size_t size)
{
  cudaSyscallStructure strce_to_send, rcvd_strce;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  memset(&rcvd_strce, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocManagedMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = localPtr;
  strce_to_send.syscall_type.cuda_memcpy.source = remotePtr;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = cudaMemcpyDeviceToHost;

  // send the structure
  JASSERT(write(skt_master, &strce_to_send, sizeof(strce_to_send)) != -1)
         (JASSERT_ERRNO);

  // get the payload: part of the GPU computation actually
  // XXX: We read a page at a time
  JASSERT(dmtcp::Util::readAll(skt_master, localPtr, size) == size)
         (JASSERT_ERRNO);

  // TODO: Verify the return val
  cudaError_t ret_val;
  JASSERT(read(skt_master, &ret_val, sizeof(int)) != -1)(JASSERT_ERRNO);
  JASSERT(ret_val == cudaSuccess)(ret_val)
          .Text("Failed to receive UVM data");
  JASSERT(read(skt_master, &rcvd_strce, sizeof(rcvd_strce)) != -1)
          (JASSERT_ERRNO);
}

/*
 * This gets called on a write fault. The shadow region containing the faulting
 * page is marked as dirty. The dirty pages are flushed later at a synchronization
 * point (see flushDirtyPages).
 */
static void
markDirtyRegion(void *page)
{
  dmtcp::vector<ShadowRegion>::iterator it;
  for (it = allShadowRegions().begin(); it != allShadowRegions().end(); it++) {
    if (it->addr == page) {
      it->dirty = true;
      haveDirtyPages = true;
      return;
    }
  }
}

/* NOTE: This function sends all the dirty pages to the proxy process. This
 * gets called at a "synchronization point", for example, on a call to
 * cudaLaunchKernel.
 */
void
flushDirtyPages()
{
  if (!haveDirtyPages) return;

  JTRACE("Flushing all dirty pages");
  dmtcp::vector<ShadowRegion>::iterator it;
  for (it = allShadowRegions().begin(); it != allShadowRegions().end(); it++) {
    if (it->dirty) {
      JTRACE("Send data to proxy")((void*)it->addr)(it->len);
      sendDataToProxy(it->addr, it->addr, it->len);
      // NOTE: We re-register the dirty page because UFFDIO_COPY
      //       unregisters the page.
#ifdef USERFAULTFD
      reregister_page(it->addr, it->len);
#else
      // Reset the permissions on the pages.
      JASSERT(mprotect(it->addr, it->len, PROT_NONE) == 0)
             (JASSERT_ERRNO)(it->addr).Text("Could not reset perms on page");
#endif
      it->dirty = false;
    }
  }
  haveDirtyPages = false;
}

/*
 * Creates shadow pages that are monitored for reads and writes
 * by the page fault handler.
 */
void*
create_shadow_pages(size_t size, cudaSyscallStructure *remoteInfo)
{
  int npages = size / page_size + 1;
  void *remoteAddr = remoteInfo->syscall_type.cuda_malloc.pointer;
#ifdef USERFAULTFD
  void *addr = mmap(remoteAddr, npages * page_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
#else
  // Ideally, we should start with PROT_WRITE, but PROT_WRITE also gives the
  // page read permissions. So, for safety and simplicity, we start with
  // PROT_NONE.
  void *addr = mmap(remoteAddr, npages * page_size, PROT_NONE, // | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
#endif

  JASSERT(addr != MAP_FAILED)(remoteAddr)(JASSERT_ERRNO);
  monitor_pages(addr, npages * page_size, remoteInfo);
  return addr;
}

#ifdef USERFAULTFD
static void*
fault_handler_thread(void *arg)
{
  static struct uffd_msg msg;   /* Data read from userfaultfd */
  static int fault_cnt = 0;     /* Number of faults so far handled */
  long uffd;                    /* userfaultfd file descriptor */
  static void *page_addr = NULL;
  struct uffdio_copy uffdio_copy;
  ssize_t nread;

  uffd = (long) arg;

  /* Create a page that will be copied into the faulting region */
  if (page == NULL) {
    page = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    JASSERT(page != MAP_FAILED)(JASSERT_ERRNO);
  }

  /* Loop, handling incoming events on the userfaultfd
     file descriptor */

  for (;;) {
    struct pollfd pollfd;
    int nready;
    pollfd.fd = uffd;
    pollfd.events = POLLIN;
    nready = poll(&pollfd, 1, -1);
    JASSERT(nready != -1)(JASSERT_ERRNO);

    JTRACE("fault_handler_thread():");
    JTRACE("    poll() returns") (nready)
          ((pollfd.revents & POLLIN) != 0)
          ((pollfd.revents & POLLERR) != 0);

    /* Read an event from the userfaultfd */

    nread = read(uffd, &msg, sizeof(msg));
    JASSERT(nread != 0).Text("EOF on uffd");
    JASSERT(nread != -1)(JASSERT_ERRNO).Text("read error on uffd");

    /* We expect only one kind of event; verify that assumption */

    JASSERT(msg.event == UFFD_EVENT_PAGEFAULT)(JASSERT_ERRNO)
           .Text("Unexpected event of uffd");

    /* Display info about the page-fault event */

    JTRACE("    UFFD_EVENT_PAGEFAULT event: ")
          (msg.arg.pagefault.flags)
          ((void*)msg.arg.pagefault.address);

    /* Copy the page pointed to by 'page' into the faulting
       region. Vary the contents that are copied in, so that it
       is more obvious that each fault is handled separately. */

    void *faultingPage = getAlignedAddress(msg.arg.pagefault.address,
                                           page_size);
#if 0
    if (shadowPageMap().find(faultingPage) == shadowPageMap().end()) {
      JASSERT(false)(faultingPage)(msg.arg.pagefault.address)
             .Text("No UVM page found for faulting address");
    } else {
#endif
      if (msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WRITE) {
        // We mark the region as dirty for flushing at a later sync point
        markDirtyRegion(faultingPage);
      } else {
#if 0
        receiveDataFromProxy(shadowPageMap()[faultingPage], page, page_size);
#else
        receiveDataFromProxy(faultingPage, page, page_size);
#endif
      }
#if 0
    }
#endif
    fault_cnt++;

    uffdio_copy.src = (unsigned long) page;

    /* We need to handle page faults in units of pages(!).
       So, round faulting address down to page boundary */

    uffdio_copy.dst = (unsigned long) msg.arg.pagefault.address &
      ~(page_size - 1);
    uffdio_copy.len = page_size;
    uffdio_copy.mode = 0;
    uffdio_copy.copy = 0;
    JASSERT(ioctl(uffd, UFFDIO_COPY, &uffdio_copy) != -1)(JASSERT_ERRNO);

    JTRACE("uffdio_copy.copy returned ")(uffdio_copy.copy);
  }
}


static void
reregister_page(void *addr, size_t len)
{
  JASSERT(munmap(addr, len) == 0)(JASSERT_ERRNO);
  void *newaddr = mmap(addr, len, PROT_READ | PROT_WRITE,
                    MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  JASSERT(newaddr != MAP_FAILED)(JASSERT_ERRNO);
  monitor_pages(addr, len);
}

// Public functions

void
unregister_all_pages()
{
  struct uffdio_range uffdio_range;

  dmtcp::vector<ShadowRegion>::iterator it;
  for (it = allShadowRegions().begin(); it != allShadowRegions().end(); it++) {
    uffdio_range.start = (uintptr_t)it->addr;
    uffdio_range.len = it->len;
    JTRACE("unregister region")(it->addr)(it->len);

    JASSERT(ioctl(uffd, UFFDIO_UNREGISTER, &uffdio_range) != -1)(JASSERT_ERRNO);
  }
}

void
register_all_pages()
{
  struct uffdio_register uffdio_register;

  dmtcp::vector<ShadowRegion>::iterator it;
  for (it = allShadowRegions().begin(); it != allShadowRegions().end(); it++) {
    /*
     * NOTE: For some reason, uffd doesn't re-register the page, without
     *       first munmaping it!  Arguably, this is a kernel bug.
     *
     * FIXME: We need to copy/restore the data on these pages
     */
    reregister_page(it->addr, it->len);
  }
}

int ufd_initialized = False;

void
userfaultfd_initialize(void)
{
  if (ufd_initialized) return;

  struct uffdio_api uffdio_api;
  pthread_t thr;      /* ID of thread that handles page faults */

  page_size = sysconf(_SC_PAGE_SIZE);

  uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);

  JASSERT(uffd != -1)(JASSERT_ERRNO);

  uffdio_api.api = UFFD_API;
  uffdio_api.features = 0;

  JASSERT(ioctl(uffd, UFFDIO_API, &uffdio_api) != -1)(JASSERT_ERRNO);

  JTRACE("ufd features")(uffdio_api.features);

  int s = pthread_create(&thr, NULL, fault_handler_thread,
                         reinterpret_cast<void*>(uffd));
  if (s != 0) {
    errno = s;
    JASSERT(s == 0)(JASSERT_ERRNO);
  }

  ufd_initialized = True;
}

void
reset_uffd(void)
{
  if (ufd_initialized) {
    JASSERT(page_size > 0);
  } else {
    return;
  }

  struct uffdio_api uffdio_api;
  int old_uffd = uffd;

  uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);

  JASSERT(uffd != -1)(JASSERT_ERRNO);

  JTRACE("Restoring uffd")(uffd)(old_uffd);

  if (uffd != old_uffd) {
    uffd = dmtcp::Util::changeFd(uffd, old_uffd);
    JASSERT(uffd == old_uffd)(JASSERT_ERRNO);
    JTRACE("Restored uffd")(uffd)(old_uffd);
  }

  uffdio_api.api = UFFD_API;
  uffdio_api.features = 0;

  JASSERT(ioctl(uffd, UFFDIO_API, &uffdio_api) != -1)(JASSERT_ERRNO);

  JTRACE("ufd features")((void*)uffdio_api.features);

  ufd_initialized = True;
}
#else
/***********************************************************
 * IMPORTANT:  This code makes the following assumptions.
 * 0.  The shadow pages of the application process will be kept loosely
 *     in sync with the managed memory pages of the proxy process.
 *     The managed memory pages act as shared memory between host and device.
 * 1.  The master (application process) is always in one of three states:
 *       (a) reading from managed memory; (b) writing to managed memory;
 *       (c) making a CUDA call
 * 2.  We can interpose on a transition among any of the states.
 * 2.a.  We can interpose on a transition to a CUDA call through our wrappers.
 *       Upon entering the CUDA call state, we remove read and write
 *       permission from all of the shadow pages in the application process.
 * 2.b. We can interpose on a transition to a read state because our
 *      segvfault_handler will note that the previous state is not "read".
 *      If the previous state was "write", then we remove write permissions
 *      from all shadow pages.
 *      For each read of a page, we then grant read permission and return from
 *      the handler.
 * 2.c. We can interpose on a transition to a write state because our
 *      segvfault_handler will note that the previous state is not "write".
 *      If the previous state was "read", then we remove read permissions
 *      from all shadow pages.
 *      For each write to a page, we then grant write permission and return
 *      from the handler.
 * 3.  There are some obvious optimizations that we could experiment with.
 *      For example, on transition to "read", we could grant read to all pages
 *      if we "learn" that the application is likely to read from all pages.
 *      Before this, we could copy all pages from proxy to application
 *      (eager policy).
 *      On transition to "write", we could grant write to all pages if we
 *      "learn" that the application is likely to write from all pages.
 *      We could defer copying from application to proxy until transitioning
 *      away from the "write" state.
 * 4.  There are some assumptions here.  In particular, we assume that
 *      eager reading doesn't change the semantics.  But the application could
 *      look at a read on a special page, in order to decide which page will
 *      next be modified by the device.  So, eagerly reading from other
 *      pages without waiting for the special page to be modifed would be
 *      a mistake.
 ***********************************************************/


// # define _GNU_SOURCE
# include <signal.h>

int segvfault_initialized = False;
void segvfault_handler(int, siginfo_t *, void *);

void
segvfault_initialize(void)
{
  if (segvfault_initialized) return;

  page_size = sysconf(_SC_PAGE_SIZE);
  // We need SHIFT inside segvfault_handler()
  for (SHIFT = 0; (1 << SHIFT) < page_size; SHIFT++) ;
  JASSERT(1 << SHIFT == page_size);

  // first install a PAGE FAULT handler: using sigaction
  static struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = segvfault_handler;
  sigemptyset(&action.sa_mask);

  JASSERT(sigaction(SIGSEGV, &action, NULL) != -1)
         (JASSERT_ERRNO).Text("Could not set up the segfault handler");

  segvfault_initialized = True;
}

void
segvfault_handler(int signum, siginfo_t *siginfo, void *context)
{
  static int fault_cnt = 0;     /* Number of faults so far handled */
  // get which address segfaulted
  void *addr = (void *) siginfo->si_addr;
  if (addr == NULL){
    JASSERT(false).Text("NULL address for segfault");
  }
  void *page_addr = (void *)(((long long unsigned)addr >> SHIFT) << SHIFT);
  void *faultingPage = getAlignedAddress((uintptr_t)addr, page_size);

  // Find out if this is a write fault.  (Otherwise, it's a read fault.)
  JASSERT(siginfo->si_code == SEGV_ACCERR);
  // This may be Intel-specific.  It depends on REG_ERR.
  int err = ((ucontext_t*)context)->uc_mcontext.gregs[REG_ERR];
  // FIXME:  On StackOverflow, there seem to be contradictory answers
  //   on whether we want the negation or not, below:
  //   https://stackoverflow.com/questions/17671869/how-to-identify-read-or-write-operations-of-page-fault-when-using-sigaction-hand
  // bool is_write_fault = !(err & 0x2);
  // XXX: Rohan: It seems like for write faults, we don't need the negation
  bool is_write_fault = (err & 0x2);

  // make sure the page is mapped.
  // int prot = PROT_NONE;
  // FIXME:
  //   For now, I am giving all permissions.  We need to examine
  //   the state ("read", "write", or "CUDA call"), and give permission
  //   only acccording to the required state.
  // int flags = MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS;
  // if (mmap(page_addr, page_size, prot, flags, -1, 0) == (void *)-1) {
  //   perror("mmap");
  //   exit(1);
  // }

  // FIXME:  Add: enum application_state: READ, WRITE, CUDA_CALL.
  //   Then add logic:   if (is_write_fault && application_state != WRITE) ...;
  //   and so on.

  if (is_write_fault) {
    // We mark the region as dirty for flushing at a later sync point
    markDirtyRegion(faultingPage);
    // change the permission in the corresponding mem region.
    // FIXME (see above)

    // NOTE: WRITE permissions also gives READ permissions. So, there could
    // be an issue where an application writes and then reads, for example, in
    // case of an application that writes to a page and then waits for an
    // already running kernel to update the page.
    JASSERT(mprotect(faultingPage, page_size, PROT_WRITE) == 0)
           (JASSERT_ERRNO)(addr).Text("Could not add write perms to the page");
  } else {
    // change the permission in the corresponding mem region.
    // FIXME (see above)
    // XXX: Temporarily give write permissions to read in data from the proxy
    JASSERT(mprotect(faultingPage, page_size, PROT_READ | PROT_WRITE) == 0)
           (JASSERT_ERRNO)(addr).Text("Could not add read perms to the page");
    receiveDataFromProxy(faultingPage, page_addr, page_size);
    // XXX: Remove the WRITE permissions
    JASSERT(mprotect(faultingPage, page_size, PROT_READ) == 0);
  }
  fault_cnt++;

  JTRACE("    SEGV page fault: ")
        (addr)(page_addr)(page_size);

  // FIXME:  Need to copy from UVM page to shadow page.  Change one of page_addr
  // Copy second argument from proxy process into this application process
  // receiveDataFromProxy(page_addr, page_addr, page_size);
  //  if (read(skt_live_migrate, addr, page_size) == -1){
  //    perror("read");
  //    exit(1);
  //  }
  // the execution continues where it segfaulted, it
  // reexecutes the same instruction but it won't segfault this time.
}
#endif
