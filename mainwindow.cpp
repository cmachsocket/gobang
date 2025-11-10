#include "mainwindow.h"
#include "checkerboard.h"
#include "ui_mainwindow.h"

#include<QMessageBox>
#include<QDebug>
#include <QVariant>

QPushButton *MainWindow::buttons[MAX_ROW + 5][MAX_COL + 5];
QLabel *MainWindow::_status;
QButtonGroup *MainWindow::btn_group = new QButtonGroup();
QFutureWatcher<std::pair<int, int> > *MainWindow::watcher = new QFutureWatcher<std::pair<int, int> >();
QVBoxLayout *MainWindow::vBoxLayout;
QHBoxLayout *MainWindow::hBoxLayout[MAX_ROW];

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
      , ui(new Ui::MainWindow) {
    ui->setupUi(this);
    setFixedSize(1000, 1200);
    //this->setStyleSheet("QMainWindow {background-color:rgb(255, 150, 30)}");
    vBoxLayout = new QVBoxLayout(this->centralWidget());
    checkerboard::wrapped_init();
    MainWindow::_status = new QLabel();
    btn_group->setExclusive(true);
    for (int i = 0; i < MAX_ROW; i++) {
        hBoxLayout[i] = new QHBoxLayout();
        for (int j = 0; j < MAX_COL; j++) {
            buttons[i][j] = new QPushButton();
            buttons[i][j]->setStyleSheet("QPushButton{background-color:rgb(128,128,128);font-size:20pt;}");
            buttons[i][j]->setContentsMargins(0, 0, 0, 0);
            buttons[i][j]->setProperty("is_ocurred", false);
            btn_group->addButton(buttons[i][j], checkerboard::enc_id(i, j));
            hBoxLayout[i]->addWidget(buttons[i][j]);
        }
        vBoxLayout->addLayout(hBoxLayout[i]);
    }
    QObject::connect(btn_group, &QButtonGroup::idClicked, this, &MainWindow::try_add_chess);
    connect(watcher, &QFutureWatcher<std::pair<int, int> >::finished, this, &MainWindow::task_finished);
    checkerboard::player_decide();
    statusBar()->addPermanentWidget(_status);
    _status->setText(checkerboard::now_person_player() ? "你是先手" : "你是后手");
    if (!checkerboard::now_person_player()) {
        buttons[7][7]->click();
    }
}

void MainWindow::forbid_buttons() {
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            buttons[i][j]->setStyleSheet("QPushButton{background-color:rgb(128,128,128);font-size:20pt;}");
            buttons[i][j]->setEnabled(false);
        }
    }
}

void MainWindow::enable_buttons() {
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            if ((buttons[i][j]->property("is_ocurred")) == false) {
                buttons[i][j]->setEnabled(true);
            }
        }
    }
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::try_add_chess(int id) {
    //QMessageBox message(QMessageBox::NoIcon, "RUSELT", checkerboard::now_person_player()?"YOU WIN!":"YOU LOSE!");
    //message.exec();
    int row, col;
    checkerboard::dec_id(id, row, col);
    if (!checkerboard::put_chess_valid(row, col)) {
        return;
    }
    buttons[row][col]->setEnabled(false);
    buttons[row][col]->setProperty("is_ocurred", true);
    if (checkerboard::now_player() == BLACK_POS) {
        buttons[row][col]->setText(BLACK_ICON);
    } else {
        buttons[row][col]->setText(WHITE_ICON);
    }
    buttons[row][col]->setStyleSheet("QPushButton{background-color:rgb(128,228,128);font-size:20pt;}");
    checkerboard::add_chess(row, col, checkerboard::now_player());
    if (checkerboard::is_game_over(row, col)) {
        forbid_buttons();
        QMessageBox message(QMessageBox::NoIcon, "RUSELT",
                            checkerboard::now_person_player() ? "YOU WIN!" : "YOU LOSE!");
        message.exec();
        qApp->quit();
    }
    if (checkerboard::now_person_player()) {
        forbid_buttons();
        QFuture<std::pair<int, int> > future = QtConcurrent::run(checkerboard::solve_find, row, col);
        watcher->setFuture(future);
        //状态监视
        _status->setText("对方下棋中");
    } else {
        enable_buttons();
        _status->setText("我方回合");
    }
    checkerboard::change_player();
    qDebug() << "Your pos:" << row << col;
    // qDebug() << "Button clicked:" << button->text();
    // qDebug() << "Button ID:" << buttonGroup.id(button);
    // 性能优化参考 buttonGroup.addButton(button3, 3);
}

void MainWindow::task_finished() {
    int x = watcher->result().first;
    int y = watcher->result().second;
    buttons[x][y]->setEnabled(true);
    //try_add_chess(x*MAX_COL+y);
    buttons[x][y]->click();
    //buttons[x][y]->setEnabled(false);
}
