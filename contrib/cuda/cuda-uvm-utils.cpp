#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include <linux/userfaultfd.h>
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

int ufd_initialized = False;
int page_size = -1;

bool haveDirtyPages = false;

// Private global vars

static int uffd = -1;

struct ShadowRegion {
  void *addr;
  size_t len;
  bool dirty;
};

// Private functions

static void reregister_page(void *addr, size_t len);

dmtcp::map<void*, void*>&
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

static bool
sendDataToProxy(void *remotePtr, void *localPtr, size_t size)
{
  cudaSyscallStructure strce_to_send;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocManagedMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = remotePtr;
  strce_to_send.syscall_type.cuda_memcpy.source = localPtr;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = cudaMemcpyHostToDevice;

  // send the structure
  JASSERT(write(skt_master, &strce_to_send, sizeof(strce_to_send)) != -1)
         (JASSERT_ERRNO);

  // send the payload: part of the GPU computation actually
  // XXX: We send a page at a time
  JASSERT(write(skt_master, localPtr, size) != -1)(JASSERT_ERRNO);
}

static bool
receiveDataFromProxy(void *remotePtr, void *localPtr, size_t size)
{
  cudaSyscallStructure strce_to_send;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

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
}

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

static void*
fault_handler_thread(void *arg)
{
  static struct uffd_msg msg;   /* Data read from userfaultfd */
  static int fault_cnt = 0;     /* Number of faults so far handled */
  long uffd;                    /* userfaultfd file descriptor */
  static void *page = NULL;
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
    if (shadowPageMap().find(faultingPage) == shadowPageMap().end()) {
      JASSERT(false)(faultingPage)(msg.arg.pagefault.address)
             .Text("No UVM page found for faulting address");
    } else {
      if (msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WRITE) {
        // We mark the region as dirty for flushing at a later sync point
        markDirtyRegion(faultingPage);
      } else {
        receiveDataFromProxy(shadowPageMap()[faultingPage], page, page_size);
      }
    }
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
monitor_pages(void *addr, size_t size, cudaSyscallStructure *remoteInfo = NULL)
{
  struct uffdio_register uffdio_register;

  uffdio_register.range.start = (uintptr_t)addr;
  uffdio_register.range.len = size;
  uffdio_register.mode = UFFDIO_REGISTER_MODE_MISSING;

  JTRACE("register region")(addr)(size);

  JASSERT(ioctl(uffd, UFFDIO_REGISTER, &uffdio_register) != -1)(JASSERT_ERRNO);

  if (remoteInfo) {
    // Save the location and size of the shadow region
    ShadowRegion r =  {.addr = addr, .len = size, .dirty = false};
    allShadowRegions().push_back(r);
    // Save the actual UVM region on the proxy
    shadowPageMap()[addr] = remoteInfo->syscall_type.cuda_malloc.pointer;
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
      reregister_page(it->addr, it->len);
      it->dirty = false;
    }
  }
  haveDirtyPages = false;
}

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
  JASSERT(ufd_initialized && page_size > 0);

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

/*
 * Creates shadow pages that are monitored for reads and writes
 * by the page fault handler.
 */
void*
create_shadow_pages(size_t size, cudaSyscallStructure *remoteInfo)
{
  int npages = size / page_size + 1;
  void *remoteAddr = remoteInfo->syscall_type.cuda_malloc.pointer;
  void *addr = mmap(remoteAddr, npages * page_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);

  JASSERT(addr != MAP_FAILED)(remoteAddr)(JASSERT_ERRNO);
  monitor_pages(addr, npages * page_size, remoteInfo);
  return addr;
}
