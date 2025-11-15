#ifndef GOBANG_VIEWMODEL_H
#define GOBANG_VIEWMODEL_H
#include <QFutureWatcher>
#define BLACK_ICON "\u26AB"
#define WHITE_ICON "\u26AA"
class Viewmodel : public QObject {
    Q_OBJECT
signals:
    void statusTextChanged(const QString &text);

    void requestButtonClick(int row, int col);
    void requestButtonEnable(int row,int col,bool status);
    void ButtonForbid();
    void ButtonEnable();
    void setButtonOccurred(int r,int c);
    void setButtonText(int r,int c,QString t);
    void NotifyMessageBox(QString title,QString text);
public slots:
    void try_add_chess(int id);
    void to_deside_player();
public:
    QFutureWatcher<std::pair<int, int> > *watcher = new QFutureWatcher<std::pair<int, int> >();
    void task_finished();
private:
    void check_to_debug();
};


#endif //GOBANG_VIEWMODEL_H
