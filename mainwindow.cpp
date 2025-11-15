#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "viewmodel.h"
#include<QButtonGroup>
#include<QMessageBox>
#include <QVariant>

QPushButton *MainWindow::buttons[MAX_ROW ][MAX_COL ];
QLabel *MainWindow::_status;
QButtonGroup *MainWindow::btn_group = new QButtonGroup();
QGridLayout *MainWindow::layout;
Viewmodel * viewmodel;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
      , ui(new Ui::MainWindow) {
    ui->setupUi(this);
    viewmodel = new Viewmodel();
    bg_pixmap = QPixmap(":/board.jpg");
    updateBackground();
    layout = new QGridLayout();
    MainWindow::_status = new QLabel();
    btn_group->setExclusive(true);
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            buttons[i][j] = new QPushButton();
            buttons[i][j]->setFixedSize(50,50);
            buttons[i][j]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;}");
            buttons[i][j]->setFlat(true);
            buttons[i][j]->setContentsMargins(0, 0, 0, 0);
            buttons[i][j]->setProperty("is_occurred", false);
            btn_group->addButton(buttons[i][j],  i* MAX_ROW + j);
            layout->addWidget(buttons[i][j], i, j);
        }
    }
    layout->setHorizontalSpacing(19);
    layout->setVerticalSpacing(19);
    layout->setContentsMargins(14, 9, 9, 14);
    this->centralWidget()->setLayout(layout);
    statusBar()->addPermanentWidget(_status);
    connect(btn_group, &QButtonGroup::idClicked, viewmodel, &Viewmodel::try_add_chess);
    connect(viewmodel->watcher, &QFutureWatcher<std::pair<int, int> >::finished, viewmodel, &Viewmodel::task_finished);
    connect(viewmodel, &Viewmodel::statusTextChanged, this, [](const QString &t){ _status->setText(t); });
    connect(viewmodel, &Viewmodel::requestButtonClick, this, [](int r, int c){ buttons[r][c]->click(); });
    connect(viewmodel, &Viewmodel::requestButtonEnable, this, [](int r, int c,bool status){ buttons[r][c]->setEnabled(status); });
    connect(viewmodel, &Viewmodel::setButtonOccurred, this, [](int r, int c){ buttons[r][c]->setProperty("is_occurred", true); });
    connect(viewmodel, &Viewmodel::setButtonText, this, [](int r, int c,const QString &t){
        buttons[r][c]->setText(t);
        buttons[r][c]->setStyleSheet("QPushButton{background: rgba(128,128,128,0.4);border:none;font-size:20pt;}");
    });
    connect(viewmodel, &Viewmodel::NotifyMessageBox, this, [](const QString &title,const QString &text){
        QMessageBox message(QMessageBox::NoIcon, title, text);
        message.exec();
        qApp->quit();
    });
    connect(viewmodel,&Viewmodel::ButtonForbid,this,&MainWindow::forbid_buttons);
    connect(viewmodel,&Viewmodel::ButtonEnable,this,&MainWindow::enable_buttons);
    connect(this, &MainWindow::deside_player, viewmodel, &Viewmodel::to_deside_player);
    emit deside_player();
}

void MainWindow::forbid_buttons() {
    for (auto & button_col : buttons) {
        for (auto & button : button_col) {
            button->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;}");
            button->setEnabled(false);
        }
    }
}

void MainWindow::enable_buttons() {
    for (auto & button_col : buttons) {
        for (auto & button : button_col) {
            if ((button->property("is_occurred")) == false) {
                button->setEnabled(true);
            }
        }
    }
}

MainWindow::~MainWindow() {
    delete ui;
}

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
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    updateBackground();
}