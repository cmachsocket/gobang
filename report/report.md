# 实验报告

## 总体设计（主要模块划分）

本五子棋ai程序采用MVC架构,将前后端分离。主要模块划分如下：

### 1. **用户界面模块（View）**：

采用QT实现前端,负责显示棋盘、棋子以及用户交互界面。用户可以通过点击棋盘进行落子操作，并查看游戏状态。

### 2. **前后端通信模块（Controller）**：

负责处理用户输入，将用户的落子请求传递给后端AI模块，并将AI的决策结果返回给前端进行显示。

### 3. **AI决策模块（Model）**：

核心模块，负责实现五子棋AI的决策逻辑。主要包括以下子模块：
   - **评估函数模块**：用于评估当前棋盘状态，通过cuda加速计算各个位置的得分。
   - **搜索算法模块**：采用Minimax算法结合Alpha-Beta剪枝技术,通过启发式搜索进一步提高深度搜索以选择最佳落子位置的效率。
   - **API接口模块**：提供与前端通信的接口，接收用户落子信息并返回AI的决策结果。
## 详细设计（主要算法及数据结构）

### 1. Alpha-Beta剪枝算法：

- 该算法是Minimax算法的优化版本，通过在搜索过程中剪枝不必要的分支，减少计算量，提高搜索效率。

### 2. 启发式搜索:

- 结合评估函数，对当前棋盘状态进行评分，优先搜索评分较高的分支,进而更容易修改alph 和 beta 值以触发剪枝，从而提高搜索深度和决策质量。

### 3. 可行性剪枝:

- 通过限制搜索范围，仅在距离已有棋子一定范围内的位置进行搜索，减少无意义的搜索节点，提高搜索效率。

### 4. 最优性剪枝:

- 对于已经确定的必胜或必败局面,提前终止搜索,避免不必要的计算.

### 5. CUDA加速评估函数:

- 利用CUDA并行计算能力，对棋盘上所有可能的落子位置进行评分，大幅提升评估函数的计算速度，从而支持更深层次的搜索。

### 6. 数据结构:

- 链表:储存启发式搜索中需要遍历的节点,对链表进行排序以获得搜索顺序。

### 7. 估值函数:

- 根据棋盘上各个位置的棋子分布情况,对不同的棋型进行标准化统一评分,使得算法更加高效.

## 实现（关键代码，ui）

### 1. UI:

- 使用QT设计用户界面,通过透明按钮+背景图片实现棋盘和棋子的显示.相较于之间对棋盘格子进行绘制和判断点击位置,该方法大大简化了UI设计和点击事件处理.

```cpp

bg_pixmap = QPixmap(":/board.jpg");
updateBackground();
layout = new QGridLayout();
MainWindow::_status = new QLabel();
btn_group->setExclusive(true);
for (int i = 0; i < MAX_ROW; i++) {
    for (int j = 0; j < MAX_COL; j++) {
        buttons[i][j] = new QPushButton();
        buttons[i][j]->setFixedSize(50,50);
        // make the button visually transparent and borderless while keeping font size
        buttons[i][j]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;}");
        buttons[i][j]->setFlat(true);
        buttons[i][j]->setContentsMargins(0, 0, 0, 0);
        buttons[i][j]->setProperty("is_occurred", false);
         btn_group->addButton(buttons[i][j],  i* MAX_ROW + j);
         layout->addWidget(buttons[i][j], i, j);
     }
}
```

- 信号槽机制实现前后端通信,用户点击棋盘按钮后,将落子位置传递给Controller,Controller实现复杂的业务操作逻辑,并与后端通信.

### 2. 关键代码:

- Alpha-Beta剪枝算法实现:

- 最优性剪枝:在alpha_beta函数的开头,通过调用is_game_over函数检查当前局面是否已经结束,如果结束则直接返回一个极端的分值,从而避免不必要的搜索.


```cpp
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
                access_chess_list.emplace_back(i, j);
            }
        }
    }
    access_chess_list.sort(cmp);
    if (is_max > 0) {
        for (auto it: access_chess_list) {
            int i = it.first, j = it.second;
            add_chess(i, j, -person_player);
            int tmp = alpha_beta(i, j, alph, beta, depth + 1, -is_max);
            if (tmp > alph) {
                alph = tmp;

                if (depth == 1) {
                    tar_x = i, tar_y = j;
                }
            }
            del_chess(i, j, -person_player);
            if (alph >= beta) break;
        }
        access_chess_list.clear();
        return alph;
    } else {
        for (auto it: access_chess_list) {
            int i = it.first, j = it.second;
            add_chess(i, j, person_player);
            beta = std::min(beta, alpha_beta(i, j, alph, beta, depth + 1, -is_max));
            del_chess(i, j, person_player);
            if (alph >= beta) break;
        }
        access_chess_list.clear();
        return beta;
    }
}
```

- 可行性剪枝:通过board_access数组记录每个位置的可行性分值,只有当该位置的分值大于0且该位置为空时,才将其加入到access_chess_list中.

- 启发式搜索:通过对access_chess_list链表进行排序实现.容易发现,离已有棋子较近的位置更有可能形成有效棋型,因此优先搜索这些位置有助于更快触发剪枝.


```cpp
void checkerboard::add_chess(int x, int y, int ply) {
    board[x][y] = ply;
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                board_access[i][j]+=std::max(SCALE-abs(i-x)+1,SCALE-abs(j-y)+1);
            }
        }
    }
}

void checkerboard::del_chess(int x, int y, int ply) {
    for (int i = x - SCALE; i <= x + SCALE; i++) {
        for (int j = y - SCALE; j <= y + SCALE; j++) {
            if (is_inside(i, j)) {
                board_access[i][j]-=std::max(SCALE-abs(i-x)+1,SCALE-abs(j-y)+1);
            }
        }
    }
    board[x][y] = EMPTY_POS;
}
inline bool checkerboard::cmp(const std::pair<int,int> &x, const std::pair<int,int> &y) {
    return board_access[x.first][x.second]>board_access[y.first][y.second];
}
std::list<std::pair<int, int> > access_chess_list;
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            if (board_access[i][j] and put_chess_valid(i, j)) {
                access_chess_list.emplace_back(i, j);
            }
        }
    }
```

### 3. CUDA加速评估函数:

- G_evaluate:将棋盘数据传输到GPU端,通过并行计算每个空位置的得分,最后将结果汇总返回CPU端.

- CUDA核函数clac_single_pos:每个线程计算一个空位置的得分,通过检查各个方向的棋型进行评分.

- clac_extend:计算某一个方向上连续棋子的数量及其扩展可能性.它接受两个结果:
  - count:用于计算在某一方向上,从当前位置开始连续棋子的数量.
  - empty_extend_tot:用于计算在某一方向上,从当前位置开始连续棋子后的空位扩展可能性.
  它会在两个方向上的empty_extend_tot中选择较大的一个,这是因为实际上下棋的时候只会往一边下,只有一个方向的扩展是有效的.

- empty_extend:用于计算在某一方向上,从当前位置开始连续棋子后的空位扩展情况.首先,它遍历该方向上连续的棋子,计算count值.然后,如果在连续棋子后面有一个空位,它继续检查该空位后面的棋子,计算extend_count值.最后,它返回count和extend_count的值.

- 评分标准:空节点才能下棋,所以只会计算空节点的分数.对于每个空节点,如果得到了分值5,说明下在该位置可以直接形成五子连珠,如果在某个方向上得分达到4,说明需要两步就可以形成五子连珠,以此类推;如果在两个方向上都达到了4,这样的棋型也会导致直接获胜,所以会被评分为最高分.

- 降低线程创建开销:正负分的计算会重复利用同样的代码逻辑,因此在一个线程中同时计算正负分,避免了创建两倍线程的开销.

```cpp
int G_evaluate(int person_player) {
    size_t bytes = MAX_ROW * MAX_COL * sizeof(int);
    cudaMemcpyToSymbol(cuda_board, checkerboard::board, bytes);
    cudaMemcpyToSymbol(cuda_board_access, checkerboard::board_access, bytes);
    int collect_ans = 0;
    clac_single_pos<<<MAX_ROW, CUDA_GROUP>>>(-person_player); //32是一组核心数
    cudaDeviceSynchronize();

    //cudaMemcpyFromSymbol(checkerboard::check_ans, cuda_ans, bytes);
    for (int x = 0; x < MAX_ROW; x++) {
        for (int y = 0; y < MAX_COL; y++) {
            collect_ans += cuda_ans[x][y];
        }
    }

    //cudaMemcpyFromSymbol(checkerboard::check_ans, cuda_ans, bytes);
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
    int x = static_cast<int>(blockIdx.x);
    int y = static_cast<int>(threadIdx.x);
    cuda_ans[x][y] = 0;
    if (y >= MAX_COL or !cuda_board[x][y]==EMPTY_POS  or !cuda_board_access[x][y]) {
        return;
    }
    int cuda_step_x[MAX_DIRECT + 1]{0, 1, -1, 0, 1};
    int cuda_step_y[MAX_DIRECT + 1]{0, 1, 1, 1, 0};
    int tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, ply, cuda_step_x, cuda_step_y);
        if (tmp>5)tmp=5;
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];
    }
    cuda_ans[x][y] += _ans;

    //为提升性能重复利用
    tri_count = 0, _ans = 0;
    for (int i = 1; i <= MAX_DIRECT; i++) {
        int tmp = clac_extend(i, x, y, -ply, cuda_step_x, cuda_step_y);
        if (tmp>5)tmp=5;
        if (tmp >= 4)tri_count++;
        if (tri_count > 1)_ans += SCORES[MAX_SCORE];
        else _ans += SCORES[tmp];

    }
    cuda_ans[x][y] -= _ans;
}
```


## 测试结果（运行截图，性能，棋力评估）

### 1. 运行截图:

![屏幕截图_20251115_234354.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_20251115_234354.png)

![屏幕截图_20251115_234609.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_20251115_234609.png)

![屏幕截图_20251116_011112.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_20251116_011112.png)

### 2. 性能:

- 实际表现:

在截图游戏中,AI决策深度(TARGET_DEPTH)为5,决策范围半径(SCALE)为2,每次决策的速度为:448
,277 ,200 ,262 ,375 ,581 ,693 ,593 ,757 ,917 ,801 ,1204 ,1492 ,1949 ,2742 ,2562 ,3275 ,6975 ,7144 ,1936 ,7956 ,16183 ,2719 ,4575 ,2663 ,10467 ,15997 ,7228 ,6091 ,14201 ,3460 ,13375 11541 ,9379 ,8180 ,6470 ,6739 ,6473 ,2648 ,406
   (单位:ms)

平均为 4798.35 ms 

具有实际对局能力的最大深度为7,每次决策时间会来到分钟级别.大于2的决策范围几乎不具有实际意义.它们对估值函数的计算结果影响极小,但会极大增加搜索节点数,从而导致决策时间成倍增长.

- 时间复杂度:

对于alph-beta剪枝算法,由于$TARGET\_DEPTH$较小,时间复杂度近似为$O(a^{TARGET\_DEPTH})$,其中 $a$ 为子节点数; $TARGET\_DEPTH$ 为最大搜索深度.

$n$ 受剪枝的影响较大,最坏约为 $MAX\_ROW \times MAX\_COL$ , 通过启发式搜索和各种剪枝和$SCALE$的约束,实际 $n$ 远远达不到上界.

对于估值函数,在不进行cuda加速优化的时候,时间复杂度为 $O(n \times MAX\_DIRECT \times 2 \times CHAIN\_LEN)$,其中 $n$ 为棋盘上空位置数; $MAX\_DIRECT$ 为方向数; $CHAIN\_LEN$ 为最大棋型长度.

$n$ 最大可达 $MAX\_ROW \times MAX\_COL$ ,实际远远达不到上界.

$CHAIN\_LEN$ 最大可以达到10(两边各有四个子的情况),实际上远远达不到上界,大于5的棋型很少.

通过cuda加速后,时间复杂度降低为 $O(MAX\_DIRECT \times 2 \times CHAIN\_LEN)$ ,提升了估值函数的计算速度,实际上内存拷贝和线程创建开销很高,通过 $perf$ 工具可以探明, 线程开销和拷贝的开销是计算的 $8$ 倍 ,主要是因为估值函数本身复杂度较小,但是调用次数较多.不过考虑到 $n$ 在一次估值中几乎一定会超过 $8$, 所以仍具有优化价值.

总时间复杂度为$(a^{TARGET\_DEPTH} \times MAX\_DIRECT \times 2 \times CHAIN\_LEN)$

- 空间复杂度:

空间复杂度开销较小.

在深度搜索剪枝中,随递归定义的链表会导致空间的开销,空间复杂度为 $O(a \times TARGET\_DEPTH)$ ,最坏可达到$O(MAX\_ROW \times MAX\_COL \times TARGET\_DEPTH)$,实际上远远达不到上界.

### 3. 棋力评估:

得益于估值函数的优秀设计,在 [gomoku.com](https://gomoku.com/zh-cn/single-player/) 的困难模式下,AI轻松获胜.

- 优势:

对于搜索算法,剪枝是最有效的优化手段,本程序通过**多种**剪枝方式,大大减少了搜索节点数,从而提升了搜索深度和决策质量. 同时启发式搜索通过优先搜索高评分节点,更容易触发剪枝,从而提升了搜索效率.

对于估值函数,对于各种棋型的评估被简化为统一的评分标准,不仅在保证估值的准确性的同时还大大提升了计算效率,也大大简化了代码逻辑,使得其易于维护.
对多方向的评分也进行了标准化的统一处理,使得代码更加简洁易懂. 同时通过cuda加速,大幅提升了估值函数的计算速度,从而支持更深层次的搜索.

- 局限性:

当存在被围住的中心空点时,这种点是无法连接出5个子得到获胜的,这时候AI依旧会对这种点进行评分,从而导致AI对局面分数的评估出现偏差,从而影响剪枝.但AI本身依赖的其实是搜索来得到最优解,除非这种点过多,否则不会对AI的棋力产生太大影响.

当遇到边界时,AI也会出现评估偏差,因为边界会影响某些棋型的形成,从而影响评分.要尽量避免鏖战,出现边界局面.

棋子变多时,需要计算的节点数会大幅增加,从而导致AI的决策时间上升.当增长到一定程度时,棋盘上能下的位置又会大幅减少,又会导致AI的决策时间下降.

alph-beta剪枝本身依赖于全局变量 alph 和 beta ,在多线程环境下会导致大量的互斥,影响效率,因此搜索只能单线程运行,无法利用多核CPU和GPU的优势.