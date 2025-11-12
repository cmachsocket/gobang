#pragma once
#include<algorithm>
#include <list>
#ifndef CHECKERBOARD_H
#define CHECKERBOARD_H

#define MAX_ROW 15
#define MAX_COL 15
#define INF (0x7fffffff)

#define EMPTY_POS 0
#define BLACK_POS 1
#define WHITE_POS (-1)
#define TARGET_DEP 5
#define SCALE 2
#define TIME_LOSE 10
#define MAX_SCORE 5

#define MAX_DIRECT 4
#define EMPTY_SELF 100
#define EMPTY_EXTEND 1000

const int scores[5 + 1]{0, 1, 10, 100, 1000, 10000};

class checkerboard {
public:
    static void player_decide();

    static void add_chess(int, int, int);

    static bool put_chess_valid(int, int);

    static void change_player();

    static bool is_game_over(int, int);

    static bool now_person_player();

    static int now_player();

    static std::pair<int, int> solve_find(int, int);

    static int enc_id(int, int);

    static void dec_id(int, int &, int &);
    static void wrapped_init();
    static int board[MAX_ROW ][MAX_COL ];
    static int board_access[MAX_ROW][MAX_COL];
    static int check_ans[MAX_ROW][MAX_COL];
private:
    checkerboard();

    static int alpha_beta(int, int, int, int, int, int);

    //static int depth,is_max;

    static int player, person_player, tar_x, tar_y;
    static int step_x[MAX_DIRECT + 1];
    static int step_y[MAX_DIRECT + 1];
    static void del_chess(int, int, int);
    static bool cmp(std::pair<int,int>,std::pair<int,int>);
    static int extend_line(int direct, int _player, int x, int y);
    inline static bool is_inside(int, int);
};

#endif // CHECKERBOARD_H
