//
// Created by cmach_socket on 2025/11/16.
//

#include "libcpu.h"
#include "checkerboard.h"
struct pair {
    int first,second;
};
int step_x[MAX_DIRECT + 1]{0, 1, -1, 0, 1};
int step_y[MAX_DIRECT + 1]{0, 1, 1, 1, 0};
#define SELF_SCORE 1

constexpr int SCORES[6]{0, 1, 10, 100, 1000, 10000};

inline bool is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}
inline pair empty_extend(int direct, int _player, int x, int y) {
    int count = 0, extend_count = 0;
    for (;is_inside(x, y) and checkerboard::board[x][y] == _player;
           count++, x += step_x[direct], y += step_y[direct]) {
           }
    if (is_inside(x, y) and checkerboard::board[x][y] == EMPTY_POS) {
        //可以继续从空点扩展,前提是扩展后有对应棋子
        for (x += step_x[direct], y += step_y[direct];
             is_inside(x, y) and checkerboard::board[x][y] == _player;
             extend_count++, x += step_x[direct], y += step_y[direct]) {
             }
        if (count or extend_count)extend_count++;
    }
    return {count, extend_count};
}



inline int clac_extend(int direct, int x, int y, int ply) {
    auto [count_1,empty_extend_tot_1] = empty_extend(direct, ply, x + step_x[direct], y + step_y[direct]
                                                     );
    step_x[direct] = -step_x[direct], step_y[direct] = -step_y[direct]; //改变方向

    auto [count_2,empty_extend_tot_2] = empty_extend(direct, ply, x + step_x[direct], y + step_y[direct]
                                                     );
    return count_1 + count_2 + SELF_SCORE + std::max(empty_extend_tot_1, empty_extend_tot_2); //决定扩展方向
}


inline int clac_single_pos(int x, int y, int ply) {
    int tri_count = 0, attack_score = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply);
        if (tmp > 5) tmp = 5;
        if (tmp >= 4) tri_count++;
        if (tri_count > 1) attack_score += SCORES[MAX_SCORE];
        else attack_score += SCORES[tmp];
    }

    tri_count = 0;
    int defend_score = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, -ply);
        if (tmp > 5) tmp = 5;
        if (tmp >= 4) tri_count++;
        if (tri_count > 1) defend_score += SCORES[MAX_SCORE];
        else defend_score += SCORES[tmp];
    }

    return attack_score - defend_score;
}

int G_evaluate(int person_player) {
    int collect_ans = 0;
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans += clac_single_pos(x, y, person_player);
        }
    }
    return collect_ans;
}


