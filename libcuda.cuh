#pragma once
#ifndef LIBCUDA_H
#define LIBCUDA_H
#include<cuda_runtime.h>
#define CUDA_GROUP 32

struct pair {
    int first,second;
};

__device__ __inline__ bool cuda_is_inside(int x, int y);
__device__ __inline__ pair empty_extend(int direct,int _player, int x, int y,const int cuda_step_x[],const int cuda_step_y[]);
__device__ __inline__ int clac_extend(int direct, int x, int y , int ply,int cuda_step_x[],int cuda_step_y[]);
__global__ void clac_single_pos(int ply);

extern "C"
int G_evaluate(int person_player) ;

#endif //LIBCUDA_H