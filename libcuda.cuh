#pragma once
#ifndef LIBCUDA_H
#define LIBCUDA_H
// When compiling with NVCC (__CUDACC__ defined) enable CUDA attributes for function declarations.
#include "checkerboard.h"
#include <cuda_runtime.h>
#ifdef __CUDACC__
  #define CUDA_DEVICE __device__
  #define CUDA_GLOBAL __global__
  #define CUDA_INLINE __inline__
#else
  #define CUDA_DEVICE
  #define CUDA_GLOBAL
  #define CUDA_INLINE inline
#endif

// Header should only declare extern globals; the __managed__ definitions live in the .cu file.
__managed__ int board[MAX_ROW][MAX_COL];
__managed__ int board_access[MAX_ROW][MAX_COL];
__managed__ int ans[MAX_ROW][MAX_COL];
__managed__ int cuda_step_x[MAX_DIRECT + 1];
__managed__ int cuda_step_y[MAX_DIRECT + 1];

// Declare device_scores as extern __constant__ when compiling with NVCC. Define it in cuda.cu.
#ifdef __CUDACC__
  extern __constant__ int device_scores[6]{0, 1, 10, 100, 1000, 10000};
  #define SCORES device_scores
#else
  #define SCORES scores
#endif
extern "C"
void init();
CUDA_DEVICE CUDA_INLINE bool cuda_is_inside(int x, int y);
CUDA_DEVICE CUDA_INLINE int empty_extend(int direct,int _player, int x, int y);
CUDA_DEVICE CUDA_INLINE int clac_extend(int direct, int x, int y , int ply);
CUDA_GLOBAL void clac_single_pos(int ply);
extern "C"
int G_evaluate(int person_player) ;

#endif //LIBCUDA_H