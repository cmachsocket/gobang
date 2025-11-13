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
QGridLayout *MainWindow::layout;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
      , ui(new Ui::MainWindow) {
    ui->setupUi(this);
    // Load the original pixmap into the member and apply a stretched background
    bg_pixmap = QPixmap(":/board.jpg");
    // perform initial background setup (will scale to fill the window without preserving aspect ratio)
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
            buttons[i][j]->setProperty("is_ocurred", false);
            btn_group->addButton(buttons[i][j], checkerboard::enc_id(i, j));
            layout->addWidget(buttons[i][j], i, j);
        }
    }
    // set fixed spacing between grid cells to 11 (horizontal and vertical)
    layout->setHorizontalSpacing(19);
    layout->setVerticalSpacing(19);
    layout->setContentsMargins(14, 9, 9, 14);
    this->centralWidget()->setLayout(layout);
    qDebug() << this->width() << this->height();//2
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
            // keep buttons transparent and borderless; show a disabled-looking text color
            buttons[i][j]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;color:rgb(255,255,255);}");
            buttons[i][j]->setEnabled(false);
        }
    }
}

void MainWindow::enable_buttons() {
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            if ((buttons[i][j]->property("is_ocurred")) == false) {

                buttons[i][j]->setText(QString::number(checkerboard::check_ans[i][j]));
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
    } else if (checkerboard::now_player() == WHITE_POS){
        buttons[row][col]->setText(WHITE_ICON);
    }

    // ensure placed piece remains on a transparent, borderless button
    buttons[row][col]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;}");
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

// Add implementations for updateBackground and resizeEvent to enable stretch-fill behavior
void MainWindow::updateBackground() {
    if (bg_pixmap.isNull()) {
        return;
    }
    QSize targetSize = this->centralWidget() ? this->centralWidget()->size() : this->size();
    QPixmap scaled = bg_pixmap.scaled(targetSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(scaled));
    if (this->centralWidget()) {
        this->centralWidget()->setAutoFillBackground(true);
        this->centralWidget()->setPalette(palette);
    } else {
        this->setAutoFillBackground(true);
        this->setPalette(palette);
    }
    //this->setFixedSize(600,600);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    updateBackground();
}
