#include "libcuda.cuh"
#include<stdio.h>
// Define device symbols (must match extern declarations in header)

__global__ void init() {
    cuda_step_x[0] = 0;
    cuda_step_x[1] = 1;
    cuda_step_x[2] = -1;
    cuda_step_x[3] = 0;
    cuda_step_x[4] = 1;
    cuda_step_y[0] = 0;
    cuda_step_y[1] = 1;
    cuda_step_y[2] = 1;
    cuda_step_y[3] = 1;
    cuda_step_y[4] = 0;
}

void cuda_init() {
    init<<<1,1>>>();
    cudaDeviceSynchronize();
}

int G_evaluate(int person_player) {
    size_t bytes = MAX_ROW * MAX_COL * sizeof(int);
    cudaMemcpyToSymbol(cuda_board, checkerboard::board, bytes);
    cudaMemcpyToSymbol(cuda_board_access, checkerboard::board_access, bytes);
    int collect_ans = 0;
    //printf("%d nn\n",cuda_board[7][7]);
    clac_single_pos<<<MAX_ROW, 32>>>(-person_player);
    cudaDeviceSynchronize();
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            //collect_ans+=checkerboard::evaluate_ans[x][y];
            collect_ans += cuda_ans[x][y];
        }
    }

    cudaMemcpyFromSymbol(checkerboard::check_ans, cuda_ans, bytes);
    return collect_ans;
}

__device__ __inline__ bool cuda_is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

__device__ __inline__ int empty_extend(int direct, int _player, int x, int y) {
    int count = 0, did_extend = 0;
    // for (int i=1;i<=4;i++) {
    //     printf("%d %d\n",cuda_step_x[i],cuda_step_y[i]);
    // }
    for (; cuda_is_inside(x, y) and cuda_board[x][y] == _player;
           count++, x += cuda_step_x[direct], y += cuda_step_y[direct], did_extend = 1) {
    }
    if (cuda_is_inside(x, y)) {
        //这里有两种情况：自身就是空点，扩展到一个空点
        if (cuda_board[x][y] == EMPTY_POS) {
            count += EMPTY_SELF;
        } else if (!did_extend and cuda_board[x][y] == -_player) {
            count += EMPTY_EXTEND;
        }
    }
    //if (count)printf("%d %d %d %d\n",x,y,_player,count);
    return count;
}

__device__ __inline__ int clac_extend(int direct, int x, int y, int ply) {
    int count = 0, is_empty = 0, empty_extend_tot = 0;
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    cuda_step_x[direct] = -cuda_step_x[direct], cuda_step_y[direct] = -cuda_step_y[direct]; //改变方向
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    empty_extend_tot = count / EMPTY_EXTEND;
    count = count % EMPTY_EXTEND;
    is_empty = count / EMPTY_SELF;
    count = count % EMPTY_SELF;
    if (is_empty)count++;
    //if (count)printf("%d %d %d | %d %d %d %d\n",x,y,ply,count,empty_extend_tot,is_empty,count+empty_extend_tot);
    return count + empty_extend_tot;
}

__global__ void clac_single_pos(int ply) {
    int x = static_cast<int>(blockIdx.x);
    int y = static_cast<int>(threadIdx.x);
    cuda_ans[x][y] = 0;
    // if (cuda_board[x][y]==EMPTY_POS and cuda_board_access[x][y]) {
    //     printf("A");
    // }
    if (y >= MAX_COL or cuda_board[x][y] != EMPTY_POS or !cuda_board_access[x][y]) {
        return;
    }

    //printf("%d %d %d\n",x,y,cuda_board_access[x][y]);
    int tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply);
        //if (tmp)printf("%d %d %d %d\n",x,y,ply,tmp);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];
    }
    cuda_ans[x][y] = _ans;

    //为提升性能重复利用
    tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, -ply);
        //if (tmp)printf("%d %d %d %d\n",x,y,ply,tmp);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];

        //printf("%d %d %d\n",x,y,tmp);
    }
    cuda_ans[x][y] -= _ans;
    //printf("FF");
}
