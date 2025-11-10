#include "checkerboard.h"
#include "libcuda.cuh"

// Definitions of managed globals so they are accessible on both host and device.
__managed__ int** board;
__managed__ int** board_access;
__managed__ int** ans;
__managed__ int *cuda_step_x ;
__managed__ int *cuda_step_y ;

void init() {
    cudaMallocManaged(&board, sizeof(int*) * MAX_ROW);
    cudaMallocManaged(&board_access, sizeof(int*) * MAX_ROW);
    cudaMallocManaged(&ans, sizeof(int*) * MAX_ROW);
    cudaMallocManaged(&cuda_step_x, sizeof(int) * (MAX_DIRECT + 1));
    cudaMallocManaged(&cuda_step_y, sizeof(int) * (MAX_DIRECT + 1));
    cuda_step_x[0] = 0; cuda_step_x[1] = 1; cuda_step_x[2] = -1; cuda_step_x[3] = 0; cuda_step_x[4] = 1;
    cuda_step_y[0] = 0; cuda_step_y[1] = 1; cuda_step_y[2] = 1; cuda_step_y[3] = 1; cuda_step_y[4] = 0;
    for (int i = 0; i < MAX_ROW; i++) {
        cudaMallocManaged(&board[i], sizeof(int) * MAX_COL);
        cudaMallocManaged(&board_access[i], sizeof(int) * MAX_COL);
        cudaMallocManaged(&ans[i], sizeof(int) * MAX_COL);
    }
}


int G_evaluate(int person_player) {
    int collect_ans = 0;
    clac_single_pos<<<MAX_ROW, MAX_ROW>>>(-person_player);
    cudaDeviceSynchronize();
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans+=ans[x][y];
        }
    }
    clac_single_pos<<<MAX_ROW, MAX_COL>>>(-person_player);
    cudaDeviceSynchronize();
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans-=ans[x][y];
        }
    }
    return collect_ans;
}

__device__ __inline__ bool cuda_is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

__device__ __inline__ int empty_extend(int direct,int _player, int x, int y) {
    int count = 0, did_extend = 0;
    for (; cuda_is_inside(x, y) and board[x][y] == _player;
           count++, x += cuda_step_x[direct], y += cuda_step_y[direct], did_extend = 1) {
           }
    if (cuda_is_inside(x, y) and (board[x][y] == EMPTY_POS or (!did_extend and board[x][y] == -_player))) {
        //这里有两种情况：自身就是空点，扩展到一个空点
        count += EMPTY_NUM;
    }
    return count;
}

__device__ __inline__ int clac_extend(int direct, int x, int y, int ply) {
    int count = 0, empty_tot = 0;
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    cuda_step_x[direct] = -cuda_step_x[direct], cuda_step_y[direct] = -cuda_step_y[direct]; //改变方向
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    empty_tot = count / EMPTY_NUM;
    count = count % EMPTY_NUM;
    //assert(count+empty_tot<=5);
    return count + empty_tot;
}

__global__ void clac_single_pos(int ply) {

    int x=blockIdx.x;
    int y=threadIdx.y;
    ans[x][y]=0;
    if (board[x][y] != EMPTY_POS) {
        return ;
    }
    int tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += scores[MAX_SCORE];
        else _ans += scores[tmp];
    }
    ans[x][y]=_ans;
}
