#include "checkerboard.h"

#include <list>
#include<QDebug>
#include <stdlib.h>
#include "libcuda.cuh"

int checkerboard::player = BLACK_POS;
int checkerboard::person_player;
int checkerboard::step_x[MAX_DIRECT + 1] = {0, 1, -1, 0, 1};
int checkerboard::step_y[MAX_DIRECT + 1] = {0, 1, 1, 1, 0};
int checkerboard::board[MAX_ROW][MAX_COL];
int checkerboard::board_access[MAX_ROW][MAX_COL];
int checkerboard::check_ans[MAX_ROW][MAX_COL];
//int checkerboard::depth=1;
//int checkerboard::is_max=1;
int checkerboard::tar_x = 0;
int checkerboard::tar_y = 0;

checkerboard::checkerboard() = default;



void checkerboard::player_decide() {
    srand(time(nullptr));
    person_player = rand()%2? BLACK_POS:WHITE_POS;
}

void checkerboard::add_chess(int x, int y, int ply) {
    board[x][y] = ply;
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                board_access[i][j]+=std::max(abs(i-x),abs(j-y));
            }
        }
    }
}

void checkerboard::del_chess(int x, int y, int ply) {
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                //内存泄漏！！！
                board_access[i][j]-=std::max(abs(i-x),abs(j-y));;
            }
        }
    }
    board[x][y] = EMPTY_POS;
}

bool checkerboard::put_chess_valid(int x, int y) {
    return board[x][y] == EMPTY_POS;
}

void checkerboard::change_player() {
    player = -player;
}

bool checkerboard::now_person_player() {
    return player == person_player;
}

bool checkerboard::is_game_over(int x, int y) {
    if (x == -1 and y == -1)return false;
    int ply = board[x][y];
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int count = 1;
        count += extend_line(i, ply, x + step_x[i], y + step_y[i]);
        step_x[i] = -step_x[i], step_y[i] = -step_y[i]; //改变方向
        count += extend_line(i, ply, x + step_x[i], y + step_y[i]);
        if (count >= 5) {
            return true;
        }
    }
    return false;
}

inline int checkerboard::extend_line(int direct, int _player, int x, int y) {
    int count = 0;
    for (; is_inside(x, y) and board[x][y] == _player; count++, x += step_x[direct], y += step_y[direct]) {
    }
    return count;
}

inline bool checkerboard::is_inside(int x, int y) {
    return ((x >= 0 and x < MAX_ROW) and (y >= 0 and y < MAX_COL));
}

int checkerboard::now_player() {
    return player;
}

std::pair<int, int> checkerboard::solve_find(int x, int y) {
    //depth=0,is_max=1;
    qDebug() << alpha_beta(-1, -1, -INF,INF, 1, 1);


    return std::make_pair(tar_x, tar_y);
}
inline bool checkerboard::cmp(const std::pair<int,int> &x, const std::pair<int,int> &y) {
    return board_access[x.first][x.second]>board_access[y.first][y.second];

}
int checkerboard::alpha_beta(int x, int y, int alph, int beta, int depth, int is_max) {
    if (is_game_over(x, y)) {
        return -is_max * scores[MAX_SCORE] * TIME_LOSE;
    }
    if (depth >= TARGET_DEP) {
        return G_evaluate(person_player);;

    }
    std::list<std::pair<int, int> > access_chess_list;
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            if (board_access[i][j] and put_chess_valid(i, j)) {
                //assert(i==7 and j==7 and put_chess_valid(i, j));
                access_chess_list.emplace_back(i, j);
            }
        }
    }
    access_chess_list.sort(cmp);
    if (is_max > 0) {
        for (auto it: access_chess_list) {
            int i = it.first, j = it.second;
            add_chess(i, j, -person_player);
            //is_max=!is_max;depth++;
            int tmp = alpha_beta(i, j, alph, beta, depth + 1, -is_max);
            //if (depth == 1)
            //qDebug() << tar_x << tar_y << alph << tmp << i << j;
            if (tmp > alph) {
                alph = tmp;

                if (depth == 1) {
                    tar_x = i, tar_y = j;
                }
            }
            //alph = std::max(alph, );
            //is_max=!is_max;depth--;
            del_chess(i, j, -person_player);
            if (alph >= beta) break;
        }
        access_chess_list.clear();
        //assert(alph!=1);
        return alph;
    } else {
        for (auto it: access_chess_list) {
            int i = it.first, j = it.second;
            add_chess(i, j, person_player);
            //is_max=!is_max;depth++;
            beta = std::min(beta, alpha_beta(i, j, alph, beta, depth + 1, -is_max));
            //is_max=!is_max;depth--;
            del_chess(i, j, person_player);
            if (alph >= beta) break;
            //assert(beta!=1);
        }
        access_chess_list.clear();
        return beta;
    }
}
int checkerboard::wrapped_G() { //调试代码暴露接口
    return G_evaluate(person_player);
}
