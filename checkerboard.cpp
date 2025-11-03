#include "checkerboard.h"
#include<QDebug>

int checkerboard::player = BLACK_POS;
int checkerboard::person_player;
int checkerboard::board[MAX_ROW][MAX_COL];
int checkerboard::access[MAX_ROW][MAX_COL];
int checkerboard::step_x[MAX_DIRECT + 1] = {0, 1, -1, 0, 1};
int checkerboard::step_y[MAX_DIRECT + 1] = {0, 1, 1, 1, 0};
//int checkerboard::depth=1;
//int checkerboard::is_max=1;
int checkerboard::tar_x = 0;
int checkerboard::tar_y = 0;

checkerboard::checkerboard() {
}

void checkerboard::dec_id(int id, int &row, int &col) {
    row = id / MAX_COL;
    col = id % MAX_COL;
}

int checkerboard::enc_id(int x, int y) {
    return x * MAX_ROW + y;
}

void checkerboard::player_decide() {
    person_player = WHITE_POS;
}

void checkerboard::add_chess(int x, int y, int ply) {
    board[x][y] = ply;
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                access[i][j]++;
            }
        }
    }
}

void checkerboard::del_chess(int x, int y, int ply) {
    board[x][y] = EMPTY_POS;
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                //内存泄漏！！！
                access[i][j]--;
            }
        }
    }
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

int checkerboard::empty_extend(int direct, int _player, int x, int y) {
    int count = 0, did_extend = 0;
    for (; is_inside(x, y) and board[x][y] == _player;
           count++, x += step_x[direct], y += step_y[direct], did_extend = 1) {
    }
    if (is_inside(x, y) and (board[x][y] == EMPTY_POS or (!did_extend and board[x][y] == -_player))) {
        //这里有两种情况：自身就是空点，扩展到一个空点
        count += EMPTY_NUM;
    }
    return count;
}

int checkerboard::now_player() {
    return player;
}

std::pair<int, int> checkerboard::solve_find(int x, int y) {
    //depth=0,is_max=1;
    qDebug() << alpha_beta(-1, -1, -INF,INF, 1, 1);
    return std::make_pair(tar_x, tar_y);
}

int checkerboard::alpha_beta(int x, int y, int alph, int beta, int depth, int is_max) {
    //qDebug() << "Button clicked:" ;
    //assert(alph!=1);
    //assert(board[5][10]==0);
    if (is_game_over(x, y)) {
        return -is_max * scores[MAX_SCORE] * TIME_LOSE;
    }
    if (depth >= TARGET_DEP) {
        return G_evaluate();
    }

    if (is_max > 0) {
        for (int i = 0; i < MAX_ROW; i++) {
            for (int j = 0; j < MAX_COL; j++) {
                if (access[i][j] and put_chess_valid(i, j)) {
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
                }
                if (alph >= beta) break;
            }
        }
        //assert(alph!=1);
        return alph;
    } else {
        for (int i = 0; i < MAX_ROW; i++) {
            for (int j = 0; j < MAX_COL; j++) {
                if (access[i][j] and put_chess_valid(i, j)) {
                    add_chess(i, j, person_player);
                    //is_max=!is_max;depth++;
                    beta = std::min(beta, alpha_beta(i, j, alph, beta, depth + 1, -is_max));
                    //is_max=!is_max;depth--;
                    del_chess(i, j, person_player);
                }
                if (alph >= beta) break;
            }
        }
        //assert(beta!=1);
        return beta;
    }
}

inline int checkerboard::clac_single_pos(int x, int y, int ply) {
    int tri_count = 0, ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply);
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)ans += scores[MAX_SCORE];
        else ans += scores[tmp];
    }
    return ans;
}

int checkerboard::G_evaluate() {
    int ans = 0;
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            if (put_chess_valid(x, y)) {
                if (board[7][11]==1 and x==5 and y==9){
                    qDebug()<<"1";
                }
                ans += clac_single_pos(x, y, -person_player);
                ans -= clac_single_pos(x, y, person_player);
            }
        }
    }
    return ans;
}


int checkerboard::clac_extend(int direct, int x, int y, int ply) {
    int count = 0, empty_tot = 0;
    count += empty_extend(direct, ply, x + step_x[direct], y + step_y[direct]);
    step_x[direct] = -step_x[direct], step_y[direct] = -step_y[direct]; //改变方向
    count += empty_extend(direct, ply, x + step_x[direct], y + step_y[direct]);
    empty_tot = count / EMPTY_NUM;
    count = count % EMPTY_NUM;
    //assert(count+empty_tot<=5);
    return count + empty_tot;
}
