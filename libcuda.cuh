#pragma once
#ifndef LIBCUDA_H
#define LIBCUDA_H

// When compiling with NVCC (__CUDACC__ defined) enable CUDA attributes.
// When compiling with a host C++ compiler, these macros expand to nothing
// (or to a sensible host equivalent) so the header can be included safely.
#ifdef __CUDACC__
  #define CUDA_MANAGED __managed__
  #define CUDA_DEVICE __device__
  #define CUDA_GLOBAL __global__
  #define CUDA_INLINE __inline__
#else
  #define CUDA_MANAGED
  #define CUDA_DEVICE
  #define CUDA_GLOBAL
  #define CUDA_INLINE inline
#endif

// CUDA-managed globals are declared as extern so they are visible to both host and device code
// when compiled by NVCC; when compiled by a host compiler these are ordinary extern declarations.
extern CUDA_MANAGED int** board;
extern CUDA_MANAGED int** board_access; // renamed from _access to avoid reserved identifier
extern CUDA_MANAGED int** ans;
extern CUDA_MANAGED int *cuda_step_x ;
extern CUDA_MANAGED int *cuda_step_y ;

void init();
CUDA_DEVICE CUDA_INLINE bool cuda_is_inside(int x, int y);
CUDA_DEVICE CUDA_INLINE int empty_extend(int direct,int _player, int x, int y);
CUDA_DEVICE CUDA_INLINE int clac_extend(int direct, int x, int y , int ply);
CUDA_GLOBAL void clac_single_pos(int ply);
int G_evaluate(int person_player) ;

#endif //LIBCUDA_H