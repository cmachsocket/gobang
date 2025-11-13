#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<QPushButton>
#include<QLabel>
#include <QtConcurrent>
#include <QPixmap>
#include <QResizeEvent>
#include <QGridLayout>
#define BLACK_ICON "\u26AB"
#define WHITE_ICON "\u26AA"
#define MAX_ROW 15
#define MAX_COL 15
#define EMPTY_POS 0
#define BLACK_POS 1
#define WHITE_POS (-1)
QT_BEGIN_NAMESPACE

namespace Ui {
    class MainWindow;
}

QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT
public slots:
    void forbid_buttons();
    void enable_buttons();
signals:
    void deside_player();
public:
    static QPushButton *buttons[MAX_ROW][MAX_COL];
    static QLabel *_status;

    MainWindow(QWidget *parent = nullptr);

    ~MainWindow();

private:
    Ui::MainWindow *ui;



    static QButtonGroup *btn_group;
    static QGridLayout *layout;


    // background pixmap used for scaling and filling the window
    QPixmap bg_pixmap;
    void updateBackground();

protected:
    void resizeEvent(QResizeEvent *event) override;
};
#endif // MAINWINDOW_H
