#include "libcuda.cuh"
#include "checkerboard.h"
#define SELF_SCORE 1
__constant__ int SCORES[6]{0, 1, 10, 100, 1000, 10000};
__device__ int cuda_board[MAX_ROW][MAX_COL];
__device__ int cuda_board_access[MAX_ROW][MAX_COL];
__managed__ int cuda_ans[MAX_ROW][MAX_COL];
__managed__ int cuda_task_x[MAX_ROW * MAX_COL];
__managed__ int cuda_task_y[MAX_ROW * MAX_COL];
__managed__ int cuda_task_count;
__managed__ int cuda_task_cursor;

int G_evaluate(int person_player) {
    size_t bytes = MAX_ROW * MAX_COL * sizeof(int);
    cudaMemcpyToSymbol(cuda_board, checkerboard::board, bytes);
    cudaMemcpyToSymbol(cuda_board_access, checkerboard::board_access, bytes);

    cuda_task_count = 0;
    cuda_task_cursor = 0;
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            cuda_ans[x][y] = 0;
            if (checkerboard::board[x][y] == EMPTY_POS && checkerboard::board_access[x][y]) {
                int idx = cuda_task_count++;
                cuda_task_x[idx] = x;
                cuda_task_y[idx] = y;
            }
        }
    }

    if (cuda_task_count > 0) {
        clac_single_pos<<<MAX_ROW, CUDA_GROUP>>>(-person_player);
        cudaDeviceSynchronize();
    }

    int collect_ans = 0;
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans += cuda_ans[x][y];
        }
    }
    return collect_ans;
}

__device__ __inline__ bool cuda_is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

__device__ __inline__ pair empty_extend(int direct, int _player, int x, int y, const int cuda_step_x[], const int cuda_step_y[]) {
    int count = 0, extend_count = 0;
    for (; cuda_is_inside(x, y) and cuda_board[x][y] == _player;
           count++, x += cuda_step_x[direct], y += cuda_step_y[direct]) {

    }
    if (cuda_is_inside(x, y) and cuda_board[x][y] == EMPTY_POS) { //可以继续从空点扩展,前提是扩展后有对应棋子
        for (x += cuda_step_x[direct], y += cuda_step_y[direct];
            cuda_is_inside(x,y) and cuda_board[x][y] == _player;
            extend_count++, x += cuda_step_x[direct], y += cuda_step_y[direct]) {
        }
        if (count or extend_count)extend_count++;
    }
    return {count,extend_count};
}

__device__ __inline__ int clac_extend(int direct, int x, int y, int ply, int cuda_step_x[], int cuda_step_y[]) {
    auto [count_1,empty_extend_tot_1] = empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct], cuda_step_x, cuda_step_y);
    cuda_step_x[direct] = -cuda_step_x[direct], cuda_step_y[direct] = -cuda_step_y[direct]; //改变方向

    auto [count_2,empty_extend_tot_2] = empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct], cuda_step_x, cuda_step_y);
    return count_1 + count_2 + SELF_SCORE + max(empty_extend_tot_1 , empty_extend_tot_2);//决定扩展方向

}

__global__ void clac_single_pos(int ply) {
    while (true) {
        int task_idx = atomicAdd(&cuda_task_cursor, 1);
        if (task_idx >= cuda_task_count) {
            break;
        }
        int x = cuda_task_x[task_idx];
        int y = cuda_task_y[task_idx];

        int attack_step_x[MAX_DIRECT + 1]{0, 1, -1, 0, 1};
        int attack_step_y[MAX_DIRECT + 1]{0, 1, 1, 1, 0};
        int tri_count = 0, attack_score = 0;
        for (int i = 1; i <= MAX_DIRECT; i++) {
            int tmp = clac_extend(i, x, y, ply, attack_step_x, attack_step_y);
            if (tmp > 5) tmp = 5;
            if (tmp >= 4) tri_count++;
            if (tri_count > 1) attack_score += SCORES[MAX_SCORE];
            else attack_score += SCORES[tmp];
        }

        int defend_step_x[MAX_DIRECT + 1]{0, 1, -1, 0, 1};
        int defend_step_y[MAX_DIRECT + 1]{0, 1, 1, 1, 0};
        tri_count = 0;
        int defend_score = 0;
        for (int i = 1; i <= MAX_DIRECT; i++) {
            int tmp = clac_extend(i, x, y, -ply, defend_step_x, defend_step_y);
            if (tmp > 5) tmp = 5;
            if (tmp >= 4) tri_count++;
            if (tri_count > 1) defend_score += SCORES[MAX_SCORE];
            else defend_score += SCORES[tmp];
        }

        cuda_ans[x][y] = attack_score - defend_score;
    }
}
