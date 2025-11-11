#include "libcuda.cuh"
#include<stdio.h>

// Define device symbols (must match extern declarations in header)

__global__ void init() {
    cuda_step_x[0] = 0; cuda_step_x[1] = 1; cuda_step_x[2] = -1; cuda_step_x[3] = 0; cuda_step_x[4] = 1;
    cuda_step_y[0] = 0; cuda_step_y[1] = 1; cuda_step_y[2] = 1; cuda_step_y[3] = 1; cuda_step_y[4] = 0;
}
void cuda_init() {
    init<<<1,1>>>();
    cudaDeviceSynchronize();
}
int G_evaluate(int person_player) {
    size_t bytes = MAX_ROW * MAX_COL * sizeof(int);
    cudaError_t err;

    // copy host board -> device symbol cuda_board
    err = cudaMemcpyToSymbol(cuda_board, checkerboard::board, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(cuda_board) failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    // copy host board_access -> device symbol cuda_board_access
    err = cudaMemcpyToSymbol(cuda_board_access, checkerboard::board_access, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(cuda_board_access) failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaDeviceSynchronize();
    int collect_ans = 0;
    //printf("%d nn\n",cuda_board[7][7]);
    clac_single_pos<<<MAX_ROW, MAX_COL>>>(-person_player);
    cudaDeviceSynchronize();

    // copy device cuda_ans -> host evaluate_ans
    err = cudaMemcpyFromSymbol(checkerboard::evaluate_ans, cuda_ans, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol(cuda_ans) failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans+=checkerboard::evaluate_ans[x][y];
        }
    }
    clac_single_pos<<<MAX_ROW, MAX_COL>>>(person_player);
    cudaDeviceSynchronize();

    err = cudaMemcpyFromSymbol(checkerboard::evaluate_ans, cuda_ans, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol(cuda_ans) failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans-=checkerboard::evaluate_ans[x][y];
        }
    }
    //printf("%d df\n",person_player);
    return collect_ans;
}

__device__ __inline__ bool cuda_is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

__device__ __inline__ int empty_extend(int direct,int _player, int x, int y) {
    int count = 0, did_extend = 0;
    // for (int i=1;i<=4;i++) {
    //     printf("%d %d\n",cuda_step_x[i],cuda_step_y[i]);
    // }
    for (; cuda_is_inside(x, y) and cuda_board[x][y] == _player;
           count++, x += cuda_step_x[direct], y += cuda_step_y[direct], did_extend = 1) {

           }
    if (cuda_is_inside(x, y) and (cuda_board[x][y] == EMPTY_POS or (!did_extend and cuda_board[x][y] == -_player))) {
        //这里有两种情况：自身就是空点，扩展到一个空点
        count += EMPTY_NUM;
    }
    //if (count)printf("%d %d %d %d\n",x,y,_player,count);
    return count;
}

__device__ __inline__ int clac_extend(int direct, int x, int y, int ply) {
    int count = 0, empty_tot = 0;
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    cuda_step_x[direct] = -cuda_step_x[direct], cuda_step_y[direct] = -cuda_step_y[direct]; //改变方向
    count += empty_extend(direct, ply, x + cuda_step_x[direct], y + cuda_step_y[direct]);
    empty_tot = count / EMPTY_NUM;
    count = count % EMPTY_NUM;

    //if (count)printf("%d %d %d %d\n",x,y,ply,count);
    return count + empty_tot;
}

__global__ void clac_single_pos(int ply) {

    int x = static_cast<int>(blockIdx.x);
    int y = static_cast<int>(threadIdx.x);
    cuda_ans[x][y]=0;
    // if (cuda_board[x][y]==EMPTY_POS and cuda_board_access[x][y]) {
    //     printf("A");
    // }
    if (cuda_board[x][y] != EMPTY_POS or !cuda_board_access[x][y]) {
         return ;
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
    cuda_ans[x][y]=_ans;
    //printf("FF");
}
