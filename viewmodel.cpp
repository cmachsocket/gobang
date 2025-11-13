//
// Created by cmach_socket on 2025/11/13.
//

#include "checkerboard.h"
#include "viewmodel.h"
#include <QtConcurrent>

void Viewmodel::try_add_chess(int id) {
    //QMessageBox message(QMessageBox::NoIcon, "RESULT", checkerboard::now_person_player()?"YOU WIN!":"YOU LOSE!");
    //message.exec();
    int row=id / MAX_COL, col= id % MAX_COL;
    if (!checkerboard::put_chess_valid(row, col)) {
        return;
    }
    requestButtonEnable(row,col,false);
    emit setButtonOccurred(row,col);
    if (checkerboard::now_player() == BLACK_POS) {
        emit setButtonText (row, col, BLACK_ICON);
    } else if (checkerboard::now_player() == WHITE_POS){
        emit setButtonText (row, col, WHITE_ICON);
    }

    // ensure placed piece remains on a transparent, borderless button
    //MainWindow::buttons[row][col]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;}");
    checkerboard::add_chess(row, col, checkerboard::now_player());
    if (checkerboard::is_game_over(row, col)) {
        emit ButtonForbid();
        emit NotifyMessageBox( "RESULT",checkerboard::now_person_player() ? "YOU WIN!" : "YOU LOSE!");
    }
    if (checkerboard::now_person_player()) {
        emit ButtonForbid();
        QFuture<std::pair<int, int> > future = QtConcurrent::run(checkerboard::solve_find, row, col);
        watcher->setFuture(future);
        //状态监视
        emit statusTextChanged("对方下棋中");
    } else {
        emit ButtonEnable();
        emit statusTextChanged("我方回合");\
    }
    checkerboard::change_player();
    qDebug() << "Your pos:" << row << col;
    // qDebug() << "Button clicked:" << button->text();
    // qDebug() << "Button ID:" << buttonGroup.id(button);
    // 性能优化参考 buttonGroup.addButton(button3, 3);
}
void Viewmodel::to_deside_player() {
    checkerboard::player_decide();
    emit statusTextChanged(checkerboard::now_person_player() ? QStringLiteral("你是先手") : QStringLiteral("你是后手"));
    if (!checkerboard::now_person_player()) {
        emit requestButtonClick(7, 7);
    }
}
void Viewmodel::task_finished() {
    int x = watcher->result().first;
    int y = watcher->result().second;
    emit requestButtonEnable(x,y,true);
    //try_add_chess(x*MAX_COL+y);
    emit requestButtonClick(x, y);
    //buttons[x][y]->setEnabled(false);
}