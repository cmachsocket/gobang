#include "libcuda.cuh"
#include<stdio.h>
extern "C"
void init() {

    cuda_step_x[0] = 0; cuda_step_x[1] = 1; cuda_step_x[2] = -1; cuda_step_x[3] = 0; cuda_step_x[4] = 1;
    cuda_step_y[0] = 0; cuda_step_y[1] = 1; cuda_step_y[2] = 1; cuda_step_y[3] = 1; cuda_step_y[4] = 0;
    for (int i=0;i<MAX_ROW;i++){
        for (int j=0;j<MAX_COL;j++){
            board[i][j]=EMPTY_POS;
            board_access[i][j]=0;
            ans[i][j]=0;
        }
    }
}

extern "C"
int G_evaluate(int person_player) {
    int collect_ans = 0;
    printf("%d n\n",board[3][3]);
    clac_single_pos<<<MAX_ROW, MAX_COL>>>(-person_player);
    cudaDeviceSynchronize();
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {

            collect_ans+=ans[x][y];
        }
    }
    printf("%d 1f\n",collect_ans);
    clac_single_pos<<<MAX_ROW, MAX_COL>>>(person_player);
    cudaDeviceSynchronize();
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            //printf("%d\n",ans[x][y]);
            collect_ans-=ans[x][y];
        }
    }
    printf("%d df\n",collect_ans);
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

    int x = static_cast<int>(blockIdx.x);
    int y = static_cast<int>(threadIdx.x);
    ans[x][y]=0;
    if (board[x][y] != EMPTY_POS and !board_access[x][y]) {
        return ;
    }
    int tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];
    }
    ans[x][y]=_ans;

}
