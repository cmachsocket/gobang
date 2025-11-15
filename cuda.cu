#include "libcuda.cuh"
#include<stdio.h>
// Define device symbols (must match extern declarations in header)



int G_evaluate(int person_player) {
    size_t bytes = MAX_ROW * MAX_COL * sizeof(int);
    cudaMemcpyToSymbol(cuda_board, checkerboard::board, bytes);
    cudaMemcpyToSymbol(cuda_board_access, checkerboard::board_access, bytes);
    int collect_ans = 0;
    //printf("%d nn\n",cuda_board[7][7]);
    clac_single_pos<<<MAX_ROW, 32>>>(-person_player);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(checkerboard::check_ans, cuda_ans, bytes);
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            //collect_ans+=checkerboard::evaluate_ans[x][y];
            collect_ans += checkerboard::check_ans[x][y];
        }
    }

    //cudaMemcpyFromSymbol(checkerboard::check_ans, cuda_ans, bytes);
    //puts("!!!");
    return collect_ans;
}

__device__ __inline__ bool cuda_is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

__device__ __inline__ pair empty_extend(int direct, int _player, int x, int y, int cuda_step_x[], int cuda_step_y[]) {
    int count = 0, extend_count = 0;
    if (!cuda_is_inside(x, y) or cuda_board[x][y] != _player) {
        return {0,0};
    }
    for (; cuda_is_inside(x, y) and cuda_board[x][y] == _player;
           count++, x += cuda_step_x[direct], y += cuda_step_y[direct]) {
    }
    if (cuda_is_inside(x, y) and cuda_board[x][y] == EMPTY_POS) { //可以继续从空点扩展
        for (x += cuda_step_x[direct], y += cuda_step_y[direct],extend_count++;
            cuda_is_inside(x,y) and cuda_board[x][y] == _player;
            extend_count++, x += cuda_step_x[direct], y += cuda_step_y[direct]) {
        }
    }
    //printf("%d\n",count);
    return {count,extend_count};
}

__device__ __inline__ int clac_extend(int direct, int x, int y, int ply, int cuda_step_x[], int cuda_step_y[]) {
    //int count = 1, empty_extend_tot_1 = 0, empty_extend_tot_2 = 0;

    auto [count_1,empty_extend_tot_1] = empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct], cuda_step_x, cuda_step_y);
    cuda_step_x[direct] = -cuda_step_x[direct], cuda_step_y[direct] = -cuda_step_y[direct]; //改变方向

    auto [count_2,empty_extend_tot_2] = empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct], cuda_step_x, cuda_step_y);
    return count_1 + count_2 + 1 + max(empty_extend_tot_1 , empty_extend_tot_2);//决定扩展方向

}

__global__ void clac_single_pos(int ply) {
    int x = static_cast<int>(blockIdx.x);
    int y = static_cast<int>(threadIdx.x);
    cuda_ans[x][y] = 0;
    // if (cuda_board[x][y]==EMPTY_POS and cuda_board_access[x][y]) {
    //     printf("A");
    // }
    if (y >= MAX_COL or !cuda_board[x][y]==EMPTY_POS  or !cuda_board_access[x][y]) {
        return;
    }
    int cuda_step_x[MAX_DIRECT + 1]{0, 1, -1, 0, 1};
    int cuda_step_y[MAX_DIRECT + 1]{0, 1, 1, 1, 0};
    //printf("%d %d %d\n",x,y,cuda_board_access[x][y]);
    int tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply, cuda_step_x, cuda_step_y);
        //if (tmp)printf("%d %d %d %d\n",x,y,ply,tmp);
        if (tmp>5)tmp=5;
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];
    }
    cuda_ans[x][y] = _ans;

    //为提升性能重复利用
    tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, -ply, cuda_step_x, cuda_step_y);
        //if (tmp)printf("%d %d %d %d\n",x,y,ply,tmp);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];

        //printf("%d %d %d\n",x,y,tmp);
    }
    cuda_ans[x][y] -= _ans;
    //printf("FF");
}
