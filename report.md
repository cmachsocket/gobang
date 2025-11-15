# 实验报告

## 总体设计（主要模块划分）

本五子棋ai程序采用MVC架构,将前后端分离。主要模块划分如下：

1. **用户界面模块（View）**：采用QT实现前端,负责显示棋盘、棋子以及用户交互界面。用户可以通过点击棋盘进行落子操作，并查看游戏状态。

2. **前后端通信模块（Controller）**：负责处理用户输入，将用户的落子请求传递给后端AI模块，并将AI的决策结果返回给前端进行显示。

3. **AI决策模块（Model）**：核心模块，负责实现五子棋AI的决策逻辑。主要包括以下子模块：
   - **评估函数模块**：用于评估当前棋盘状态，通过cuda加速计算各个位置的得分。
   - **搜索算法模块**：采用Minimax算法结合Alpha-Beta剪枝技术,通过启发式搜索进一步提高深度搜索以选择最佳落子位置的效率。
   - **API接口模块**：提供与前端通信的接口，接收用户落子信息并返回AI的决策结果。
## 详细设计（主要算法及数据结构）

1. Alpha-Beta剪枝算法：
- 该算法是Minimax算法的优化版本，通过在搜索过程中剪枝不必要的分支，减少计算量，提高搜索效率。

2. 启发式搜索:
- 结合评估函数，对当前棋盘状态进行评分，优先搜索评分较高的分支,进而更容易修改alph 和 beta 值以触发剪枝，从而提高搜索深度和决策质量。

3. CUDA加速评估函数:
- 利用CUDA并行计算能力，对棋盘上所有可能的落子位置进行评分，大幅提升评估函数的计算速度，从而支持更深层次的搜索。

4. 数据结构:
- 链表:储存启发式搜索中需要遍历的节点,对链表进行排序以获得搜索顺序。

5. 估值函数:

- 根据棋盘上各个位置的棋子分布情况,对不同的棋型进行标准化统一评分,使得算法更加高效.

## 实现（关键代码，ui）

1. UI:

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

2. 关键代码:

- Alpha-Beta剪枝算法实现:

```cpp

```

## 测试结果（运行截图，性能，棋力评估）