#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "viewmodel.h"
#include<QButtonGroup>
#include<QMessageBox>
#include<QDebug>
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
    // Load the original pixmap into the member and apply a stretched background
    bg_pixmap = QPixmap(":/board.jpg");
    // perform initial background setup (will scale to fill the window without preserving aspect ratio)
    updateBackground();
    layout = new QGridLayout();
    viewmodel = new Viewmodel();
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
    // set fixed spacing between grid cells to 11 (horizontal and vertical)
    layout->setHorizontalSpacing(19);
    layout->setVerticalSpacing(19);
    layout->setContentsMargins(14, 9, 9, 14);
    this->centralWidget()->setLayout(layout);
    qDebug() << this->width() << this->height();//2
    statusBar()->addPermanentWidget(_status);
    connect(btn_group, &QButtonGroup::idClicked, viewmodel, &Viewmodel::try_add_chess);
    connect(viewmodel->watcher, &QFutureWatcher<std::pair<int, int> >::finished, viewmodel, &Viewmodel::task_finished);
    connect(viewmodel, &Viewmodel::statusTextChanged, this, [this](const QString &t){ _status->setText(t); });
    connect(viewmodel, &Viewmodel::requestButtonClick, this, [this](int r, int c){ buttons[r][c]->click(); });
    connect(viewmodel, &Viewmodel::requestButtonEnable, this, [this](int r, int c,bool status){ buttons[r][c]->setEnabled(status); });
    connect(viewmodel, &Viewmodel::setButtonOccurred, this, [this](int r, int c){ buttons[r][c]->setProperty("is_occurred", true); });
    connect(viewmodel, &Viewmodel::setButtonText, this, [this](int r, int c,QString t){ buttons[r][c]->setText(t); });
    connect(viewmodel, &Viewmodel::NotifyMessageBox, this, [this](QString title,QString text){
        QMessageBox message(QMessageBox::NoIcon, title, text);
        message.exec();
        qApp->quit();
    });
    connect(viewmodel,&Viewmodel::ButtonForbid,this,&MainWindow::forbid_buttons);
    connect(viewmodel,&Viewmodel::ButtonEnable,this,&MainWindow::enable_buttons);
    connect(this, &MainWindow::deside_player, viewmodel, &Viewmodel::to_deside_player);
    emit deside_player();
    //player_decide!!!!
    /*
     *
     *
     */
}

void MainWindow::forbid_buttons() {
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            // keep buttons transparent and borderless; show a disabled-looking text color
            //buttons[i][j]->setStyleSheet("QPushButton{background:transparent;border:none;font-size:20pt;color:rgb(255,255,255);}");
            buttons[i][j]->setEnabled(false);
        }
    }
}

void MainWindow::enable_buttons() {
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            if ((buttons[i][j]->property("is_occurred")) == false) {

                //buttons[i][j]->setText(QString::number(checkerboard::check_ans[i][j]));
                buttons[i][j]->setEnabled(true);
            }
        }
    }
}

MainWindow::~MainWindow() {
    delete ui;
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