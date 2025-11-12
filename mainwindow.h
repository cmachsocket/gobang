#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "checkerboard.h"
#include<QPushButton>
#include <QButtonGroup>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFuture>
#include<QLabel>
#include <QtConcurrent>
#include <QPixmap>
#include <QResizeEvent>
#include <QGridLayout>
#define BLACK_ICON "\u26AB"
#define WHITE_ICON "\u26AA"

QT_BEGIN_NAMESPACE

namespace Ui {
    class MainWindow;
}

QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

    ~MainWindow();

private:
    static QFutureWatcher<std::pair<int, int> > *watcher;
    Ui::MainWindow *ui;

    static void try_add_chess(int);

    static void enable_buttons();

    static QPushButton *buttons[MAX_ROW + 5][MAX_COL + 5];
    static QButtonGroup *btn_group;
    static QGridLayout *layout;
    static QLabel *_status;

    static void forbid_buttons();

    static void task_finished();

    // background pixmap used for scaling and filling the window
    QPixmap bg_pixmap;
    void updateBackground();

protected:
    void resizeEvent(QResizeEvent *event) override;
};
#endif // MAINWINDOW_H
