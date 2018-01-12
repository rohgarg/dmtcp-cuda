// Generated by cuda-wrappers.py and main.template
#ifndef _CUDA_PLUGIN_H_
#define _CUDA_PLUGIN_H_

#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_plugin.h"
#include "jassert.h"

#define PYTHON_AUTO_GENERATE 1

// for pinned memory
#define PINNED_MEM_MAX_ALLOC 100
// extern typedef struct pseudoPinnedMem pseudoPinnedMem_t;
void pseudoPinnedMem_append(void *ptr);
bool is_pseudoPinnedMem(void *ptr);
void pseudoPinnedMem_remove(void *ptr);

extern int skt_master;
extern int skt_accept;
extern int initialized;

void proxy_initialize();

enum cuda_op {
  OP_proxy_example_fnc,
  OP_proxy_cudaFree_fnc,
  OP_cudaMalloc,
  OP_cudaPointerGetAttributes,
  OP_cudaMemcpy,
  OP_cudaMemcpy2D,
  OP_cudaMallocArray,
  OP_cudaFreeArray,
  OP_cudaConfigureCall,
  OP_cudaSetupArgument,
  OP_cudaLaunch,
  OP_cudaThreadSynchronize,
  OP_cudaGetLastError,
  OP_cudaMallocPitch,
  OP_cudaDeviceReset,
  OP_cudaCreateTextureObject,
  OP_cudaPeekAtLastError,
  OP_cudaProfilerStart,
  OP_cudaProfilerStop,
  OP_cudaStreamSynchronize,
  OP_cudaUnbindTexture,
  OP_cudaDestroyTextureObject,
  OP_cudaEventQuery,
  OP_cudaDeviceCanAccessPeer,
  OP_cudaDeviceGetAttribute,
  OP_cudaDeviceSetCacheConfig,
  OP_cudaDeviceSetSharedMemConfig,
  OP_cudaDeviceSynchronize,
  OP_cudaEventCreateWithFlags,
  OP_cudaEventCreate,
  OP_cudaEventDestroy,
  OP_cudaEventSynchronize,
  OP_cudaEventElapsedTime,
  OP_cudaEventRecord,
  OP_cudaFuncGetAttributes,
  OP_cudaGetDevice,
  OP_cudaGetDeviceCount,
  OP_cudaGetDeviceProperties,
  OP_cudaMemset,
  OP_cudaSetDevice,
  OP_cudaMemcpyAsync,
  OP_cudaMemsetAsync,
  OP_cudaCreateChannelDesc,
  OP_cudaFuncSetCacheConfig,
  OP_cudaGetErrorString,
  OP_cudaMemGetInfo,
  OP_cudaStreamCreate,
  OP_cudaThreadExit,
  OP_proxy_cudaMallocManaged,
  OP_LAST_FNC
};

void FNC_proxy_example_fnc(void);
void FNC_proxy_cudaFree_fnc(void);
void FNC_cudaMalloc(void);
void FNC_cudaPointerGetAttributes(void);
void FNC_cudaMemcpy(void);
void FNC_cudaMemcpy2D(void);
void FNC_cudaMallocArray(void);
void FNC_cudaFreeArray(void);
void FNC_cudaConfigureCall(void);
void FNC_cudaSetupArgument(void);
void FNC_cudaLaunch(void);
void FNC_cudaThreadSynchronize(void);
void FNC_cudaGetLastError(void);
void FNC_cudaMallocPitch(void);
void FNC_cudaDeviceReset(void);
void FNC_cudaCreateTextureObject(void);
void FNC_cudaPeekAtLastError(void);
void FNC_cudaProfilerStart(void);
void FNC_cudaProfilerStop(void);
void FNC_cudaStreamSynchronize(void);
void FNC_cudaUnbindTexture(void);
void FNC_cudaDestroyTextureObject(void);
void FNC_cudaEventQuery(void);
void FNC_cudaDeviceCanAccessPeer(void);
void FNC_cudaDeviceGetAttribute(void);
void FNC_cudaDeviceSetCacheConfig(void);
void FNC_cudaDeviceSetSharedMemConfig(void);
void FNC_cudaDeviceSynchronize(void);
void FNC_cudaEventCreateWithFlags(void);
void FNC_cudaEventCreate(void);
void FNC_cudaEventDestroy(void);
void FNC_cudaEventSynchronize(void);
void FNC_cudaEventElapsedTime(void);
void FNC_cudaEventRecord(void);
void FNC_cudaFuncGetAttributes(void);
void FNC_cudaGetDevice(void);
void FNC_cudaGetDeviceCount(void);
void FNC_cudaGetDeviceProperties(void);
void FNC_cudaMemset(void);
void FNC_cudaSetDevice(void);
void FNC_cudaMemcpyAsync(void);
void FNC_cudaMemsetAsync(void);
void FNC_cudaCreateChannelDesc(void);
void FNC_cudaFuncSetCacheConfig(void);
void FNC_cudaGetErrorString(void);
void FNC_cudaMemGetInfo(void);
void FNC_cudaStreamCreate(void);
void FNC_cudaThreadExit(void);
void FNC_proxy_cudaMallocManaged(void);

#endif // ifndef _CUDA_PLUGIN_H_
