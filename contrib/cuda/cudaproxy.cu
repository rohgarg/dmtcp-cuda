#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>

#ifdef USE_SHM
# include <sys/ipc.h>
# include <sys/shm.h>
#endif

// Definitions of common structs shared with the main process
#include "cuda_plugin.h"

#ifdef USE_SHM
int shmID;
void *shmaddr;
#endif

int compute(int fd, cudaSyscallStructure *structure);

// TODO: The proxy should just be a child process of the main process;
//       we should get rid of this hardwired kernel
__global__ void add(int a, int b, int *c)
{
   *c = a+b;
}

__global__ void add_2_1(int a, int b, int *c)
{
	*c = a + b;
}

int main(int argc, char **argv)
{
  // set up the server
  int skt_proxy, skt_accept;
  struct sockaddr_un sa_proxy;
  char *sktname = argv[1];

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

#ifdef USE_SHM
  // read the shmID
  if (read(skt_accept, &shmID, sizeof(shmID)) == -1)
  {
    perror("read()");
    exit(EXIT_FAILURE);
  }


  if ((shmaddr = shmat(shmID, NULL, 0)) == (void *) -1)
  {
    perror("shmat()");
    exit(EXIT_FAILURE);
  }
#endif

  int return_val;
  cudaSyscallStructure structure;

  while(1)
  {
    // read the structure
    // At this stage the GPU does the computation as well.

    if (read(skt_accept, &structure, sizeof(structure)) == -1)
    {
      perror("read()");
      exit(EXIT_FAILURE);
    }
    return_val = compute(skt_accept, &structure);

    // send the result
    if (write(skt_accept, &return_val, sizeof(return_val)) == -1)
    {
      perror("write()");
      exit(EXIT_FAILURE);
    }

    // send the datastructure back
    if (write(skt_accept, &structure, sizeof(structure)) == -1)
    {
      perror("write()");
      exit(EXIT_FAILURE);
    }
   }
}


int compute(int fd, cudaSyscallStructure *structure)
{
  int return_val;
  enum cuda_syscalls op = structure->op;

  enum cudaMemcpyKind direction = (structure->syscall_type).cuda_memcpy.direction;
  size_t size = (structure->syscall_type).cuda_memcpy.size;
  void *new_source, *new_destination;

  switch (op)
  {
    //
    case CudaDeviceSync:
      return_val = cudaDeviceSynchronize();
      break;

    case CudaMalloc:
      return_val = cudaMalloc(&((structure->syscall_type).cuda_malloc.pointer),
        (structure->syscall_type).cuda_malloc.size);
      break;

    case CudaMallocManaged:
      return_val =
          cudaMallocManaged(&((structure->syscall_type).cuda_malloc.pointer),
                            (structure->syscall_type).cuda_malloc.size);
      break;

    case CudaFree:
      return_val = cudaFree((structure->syscall_type).cuda_free.pointer);
      break;

    case CudaMallocManagedMemcpy:
      switch (direction) {
        case cudaMemcpyDeviceToHost:
          // send data to the master
          if (write(fd,
                    (structure->syscall_type).cuda_memcpy.source,
                    (structure->syscall_type).cuda_memcpy.size) == -1) {
            perror("write()");
            exit(EXIT_FAILURE);
          }

          break;

        default:
          printf("bad direction value: %d\n", direction);
          exit(EXIT_FAILURE);
      }

    //
    case CudaMemcpy:
      switch(direction)
      {
        case cudaMemcpyHostToDevice:
          // receive payload

          // we need a new pointer for source
          new_source = malloc(size);
          if (new_source == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          if (read(fd, new_source, size) == -1)
          {
            perror("read()");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy((structure->syscall_type).cuda_memcpy.destination,
                                    new_source, size, direction);

          free(new_source);
          break;


        case cudaMemcpyDeviceToHost:
          // we need a new pointer for destination
          new_destination = malloc(size);
          if (new_destination == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy(new_destination,
                    (structure->syscall_type).cuda_memcpy.source, size, direction);

          // send data to the master
          if (write(fd, new_destination, size) == -1)
          {
            perror("write()");
            exit(EXIT_FAILURE);
          }

          free(new_destination);
          break;

        case cudaMemcpyDeviceToDevice:
          // GPU computation
          return_val = cudaMemcpy((structure->syscall_type).cuda_memcpy.destination,
           (structure->syscall_type).cuda_memcpy.source, size, direction);
          break;

        default:
          printf("bad direction value: %d\n", direction);
          exit(EXIT_FAILURE);

      }
      break;

    case CudaMallocArray:
      return_val = cudaMallocArray(&((structure->syscall_type).cuda_malloc_array.array),
                       &((structure->syscall_type).cuda_malloc_array.desc),
                       (structure->syscall_type).cuda_malloc_array.width,
                       (structure->syscall_type).cuda_malloc_array.height,
                       (structure->syscall_type).cuda_malloc_array.flags);

      // send shmid

      break;

    case CudaFreeArray:
      return_val = cudaFreeArray((structure->syscall_type).cuda_free_array.array);
      break;

  //  case CudaHostAlloc:
  //    return_val = cudaHostAlloc(&((structure->syscall_type).cuda_host_alloc.pHost),
  //                                (structure->syscall_type).cuda_host_alloc.size,
  //                                (structure->syscall_type).cuda_host_alloc.flags);
  //    break;

    case CudaConfigureCall:
    {
      ;
      int *gridDim = (structure->syscall_type).cuda_configure_call.gridDim;
      dim3 gDim(gridDim[0], gridDim[1], gridDim[2]);      

      int *blockDim = (structure->syscall_type).cuda_configure_call.blockDim;
      dim3 bDim(blockDim[0], blockDim[1], blockDim[2]);
      size_t sharedMem = (structure->syscall_type).cuda_configure_call.sharedMem;
      cudaStream_t stream = (structure->syscall_type).cuda_configure_call.stream;
      return_val = cudaConfigureCall(gDim, bDim, sharedMem, stream);
      
      break;
    }

    case CudaSetupArgument:
    {
      size_t size = (structure->syscall_type).cuda_setup_argument.size;
      size_t offset = (structure->syscall_type).cuda_setup_argument.offset;

      void *arg = malloc(size);
#ifdef USE_SHM
      memcpy(arg, shmaddr, size);
#endif
      if (read(fd, arg, size) == -1)
      {
        perror("read()");
        exit(EXIT_FAILURE);
      }
      return_val = cudaSetupArgument(arg, size, offset);
      break; 
    }

    case CudaLaunch:
    {
      // return_val = cudaLaunch(add);
      return_val = cudaLaunch(add_2_1);
      break; 
    }

    default:
      printf("bad op value: %d\n", (int) op);
      exit(EXIT_FAILURE);

  }
  return return_val;
}
