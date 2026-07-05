#include <QApplication>
#include <QBoxLayout>
#include <QCloseEvent>
#include <QDateTime>
#include <QFrame>
#include <QGraphicsDropShadowEffect>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QScrollBar>
#include <QTextEdit>
#include <QTimer>
#include <QWidget>

static const QString kAppDir = "/home/vicky/xiaodie/app";
static const QString kSpacemitLogo = kAppDir + "/assets/spacemit-logo.png";
static const QString kContestLogo = kAppDir + "/assets/contest-logo.jpg";

class XiaoDieWindow final : public QMainWindow {
public:
  XiaoDieWindow() {
    setWindowTitle(QStringLiteral("小蝶故事机"));
    resize(820, 456);
    setMinimumSize(760, 430);
    setStyleSheet(R"(
      QMainWindow { background: #eef4ff; }
      QLabel { color: #18202f; }
      QFrame#card { background: rgba(255,255,255,232); border: 1px solid #d7deea; border-radius: 16px; }
      QFrame#side { background: #ffffff; border: 1px solid #d7deea; border-radius: 18px; }
      QLabel#title { font-size: 25px; font-weight: 800; }
      QLabel#sub { color: #697386; font-size: 13px; }
      QLabel#section { font-size: 18px; font-weight: 800; }
      QLabel#status { color: white; font-weight: 800; border-radius: 15px; padding: 7px 12px; background: #667085; }
      QLabel#softCard { color: #344054; background: #f1f5fb; border-radius: 12px; padding: 9px 11px; font-size: 13px; }
      QPushButton { border: 0; border-radius: 10px; padding: 10px 12px; font-size: 15px; font-weight: 800; min-height: 24px; }
      QPushButton#start { background: #2563eb; color: white; }
      QPushButton#stop { background: #dc2626; color: white; }
      QPushButton#clear { background: #e8edf6; color: #253044; }
      QTextEdit { background: #f8fbff; color: #263043; border: 1px solid #d7deea; border-radius: 12px; padding: 12px; font-size: 14px; }
    )");

    auto *root = new QWidget(this);
    auto *mainLayout = new QHBoxLayout(root);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(12);
    setCentralWidget(root);

    auto *side = new QFrame(root);
    side->setObjectName("side");
    side->setFixedWidth(260);
    auto *sideLayout = new QVBoxLayout(side);
    sideLayout->setContentsMargins(18, 16, 18, 16);
    sideLayout->setSpacing(10);
    mainLayout->addWidget(side);

    spacemitLabel_ = new QLabel(side);
    spacemitLabel_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    setImage(spacemitLabel_, kSpacemitLogo, 188, 52);
    sideLayout->addWidget(spacemitLabel_);

    auto *title = new QLabel(QStringLiteral("小蝶故事机"), side);
    title->setObjectName("title");
    sideLayout->addWidget(title);

    auto *sub = new QLabel(QStringLiteral("小朋友的语音故事伙伴"), side);
    sub->setObjectName("sub");
    sideLayout->addWidget(sub);

    auto *statusCard = new QFrame(side);
    statusCard->setObjectName("card");
    auto *statusLayout = new QVBoxLayout(statusCard);
    statusLayout->setContentsMargins(14, 12, 14, 12);
    statusLayout->setSpacing(7);
    auto *statusCaption = new QLabel(QStringLiteral("小蝶状态"), statusCard);
    statusCaption->setObjectName("sub");
    statusLayout->addWidget(statusCaption);
    statusLabel_ = new QLabel(QStringLiteral("未启动"), statusCard);
    statusLabel_->setObjectName("status");
    statusLabel_->setAlignment(Qt::AlignCenter);
    statusLayout->addWidget(statusLabel_);
    detailLabel_ = new QLabel(QStringLiteral("点击开始后，按住按钮说想听的故事。"), statusCard);
    detailLabel_->setObjectName("sub");
    detailLabel_->setWordWrap(true);
    statusLayout->addWidget(detailLabel_);
    sideLayout->addWidget(statusCard);

    startButton_ = new QPushButton(QStringLiteral("开始使用"), side);
    startButton_->setObjectName("start");
    stopButton_ = new QPushButton(QStringLiteral("结束使用"), side);
    stopButton_->setObjectName("stop");
    clearButton_ = new QPushButton(QStringLiteral("清空记录"), side);
    clearButton_->setObjectName("clear");
    sideLayout->addWidget(startButton_);
    sideLayout->addWidget(stopButton_);
    sideLayout->addWidget(clearButton_);
    sideLayout->addStretch(1);

    auto *hint = new QLabel(QStringLiteral("按住小按钮说话，松开后小蝶会开始讲故事。想换一个故事时，再按一次按钮即可。"), side);
    hint->setObjectName("sub");
    hint->setWordWrap(true);
    sideLayout->addWidget(hint);

    auto *right = new QFrame(root);
    right->setObjectName("card");
    auto *rightLayout = new QVBoxLayout(right);
    rightLayout->setContentsMargins(16, 14, 16, 16);
    rightLayout->setSpacing(10);
    mainLayout->addWidget(right, 1);

    auto *top = new QVBoxLayout();
    contestLabel_ = new QLabel(right);
    contestLabel_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    setImage(contestLabel_, kContestLogo, 330, 48);
    top->addWidget(contestLabel_);
    auto *headText = new QVBoxLayout();
    auto *section = new QLabel(QStringLiteral("小蝶正在做什么"), right);
    section->setObjectName("section");
    headText->addWidget(section);
    auto *desc = new QLabel(QStringLiteral("这里会用简单的话提示当前进度。"), right);
    desc->setObjectName("sub");
    headText->addWidget(desc);
    top->addLayout(headText);
    rightLayout->addLayout(top);

    auto *guide = new QLabel(QStringLiteral("使用方式：按住按钮说出故事主题，松开后等待小蝶讲故事。"), right);
    guide->setObjectName("softCard");
    guide->setWordWrap(true);
    rightLayout->addWidget(guide);

    logView_ = new QTextEdit(right);
    logView_->setReadOnly(true);
    logView_->setLineWrapMode(QTextEdit::WidgetWidth);
    rightLayout->addWidget(logView_, 1);

    connect(startButton_, &QPushButton::clicked, this, [this]() { startService(); });
    connect(stopButton_, &QPushButton::clicked, this, [this]() { stopService(); });
    connect(clearButton_, &QPushButton::clicked, logView_, &QTextEdit::clear);

    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this, [this]() { readOutput(); });
    connect(process_, &QProcess::finished, this, [this](int code, QProcess::ExitStatus) {
      appendLog(QStringLiteral("后台服务已退出，退出码 %1").arg(code));
      setStatus(QStringLiteral("已停止"), QStringLiteral("后台进程已退出"), "#667085");
      startButton_->setEnabled(true);
    });

    appendLog(QStringLiteral("欢迎使用小蝶故事机。点击“开始使用”后，小蝶会准备好听你说话。"));
  }

protected:
  void closeEvent(QCloseEvent *event) override {
    if (process_ && process_->state() != QProcess::NotRunning) {
      process_->terminate();
      process_->waitForFinished(1200);
    }
    QMainWindow::closeEvent(event);
  }

private:
  QLabel *spacemitLabel_ = nullptr;
  QLabel *contestLabel_ = nullptr;
  QLabel *statusLabel_ = nullptr;
  QLabel *detailLabel_ = nullptr;
  QTextEdit *logView_ = nullptr;
  QPushButton *startButton_ = nullptr;
  QPushButton *stopButton_ = nullptr;
  QPushButton *clearButton_ = nullptr;
  QProcess *process_ = nullptr;
  bool storyFinishedShown_ = false;

  static void setImage(QLabel *label, const QString &path, int w, int h) {
    QPixmap pix(path);
    if (pix.isNull()) {
      label->setText(path);
      return;
    }
    label->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    label->setMinimumSize(w, h);
  }

  void setStatus(const QString &status, const QString &detail, const QString &color) {
    statusLabel_->setText(status);
    statusLabel_->setStyleSheet(QStringLiteral("color:white;font-weight:800;border-radius:15px;padding:7px 12px;background:%1;").arg(color));
    detailLabel_->setText(detail);
  }

  void appendLog(const QString &line) {
    const QString stamp = QDateTime::currentDateTime().toString("HH:mm:ss");
    logView_->append(QStringLiteral("[%1] %2").arg(stamp, line.trimmed()));
    auto *bar = logView_->verticalScrollBar();
    bar->setValue(bar->maximum());
  }

  void startService() {
    if (process_->state() != QProcess::NotRunning) {
      appendLog(QStringLiteral("小蝶已经在运行。"));
      return;
    }
    setStatus(QStringLiteral("启动中"), QStringLiteral("正在启动后台服务"), "#b7791f");
    startButton_->setEnabled(false);
    appendLog(QStringLiteral("小蝶正在准备，请稍等。"));
    process_->start(QStringLiteral("sudo"), {QStringLiteral("-n"), QStringLiteral("/usr/local/bin/xiaodie-start")});
  }

  void stopService() {
    appendLog(QStringLiteral("正在结束本次使用。"));
    if (process_->state() != QProcess::NotRunning) {
      process_->terminate();
      process_->waitForFinished(1000);
    }
    QProcess::execute(QStringLiteral("sudo"), {QStringLiteral("-n"), QStringLiteral("/usr/local/bin/xiaodie-stop")});
    setStatus(QStringLiteral("已停止"), QStringLiteral("服务已停止"), "#667085");
    startButton_->setEnabled(true);
    appendLog(QStringLiteral("小蝶已经休息了。"));
  }

  void readOutput() {
    const QString text = QString::fromUtf8(process_->readAllStandardOutput());
    for (const QString &raw : text.split('\n')) {
      const QString line = raw.trimmed();
      if (line.isEmpty()) continue;
      if (line.contains("ASR daemon starting")) {
        setStatus(QStringLiteral("准备中"), QStringLiteral("小蝶正在醒来，请稍等。"), "#b7791f");
        appendLog(QStringLiteral("小蝶正在准备听故事主题。"));
      } else if (line.contains("[asr_daemon] ready") || line.contains("ready: hold GPIO")) {
        setStatus(QStringLiteral("准备好了"), QStringLiteral("可以按住按钮说话。"), "#15935f");
        appendLog(QStringLiteral("小蝶准备好了，可以按住按钮说话。"));
        startButton_->setEnabled(false);
      } else if (line.contains("recording...")) {
        storyFinishedShown_ = false;
        setStatus(QStringLiteral("正在听"), QStringLiteral("请说出想听的故事。"), "#7c3aed");
        appendLog(QStringLiteral("小蝶正在听你说话。"));
      } else if (line.contains("recording stopped") || line.contains("recognizing speech")) {
        setStatus(QStringLiteral("正在识别"), QStringLiteral("小蝶正在听懂你刚才说的话。"), "#2563eb");
        appendLog(QStringLiteral("小蝶正在听懂你刚才说的话。"));
      } else if (line.contains("thinking story")) {
        setStatus(QStringLiteral("正在想故事"), QStringLiteral("小蝶正在想一个适合你的故事。"), "#2563eb");
        appendLog(QStringLiteral("小蝶正在想一个适合你的故事，请稍等。"));
      } else if (line.contains("DeepSeek+TTS started")) {
        storyFinishedShown_ = false;
        setStatus(QStringLiteral("正在想故事"), QStringLiteral("小蝶正在整理故事和声音。"), "#2563eb");
        appendLog(QStringLiteral("小蝶正在整理故事和声音。"));
      } else if (line.contains("ASR:")) {
        QString heard = line.section("ASR:", 1).section("elapsed=", 0, 0).trimmed();
        if (heard.isEmpty() || heard.contains("<empty>")) {
          setStatus(QStringLiteral("没听清"), QStringLiteral("可以再按住按钮说一遍。"), "#b7791f");
          appendLog(QStringLiteral("小蝶没有听清楚，可以再说一遍。"));
        } else {
          setStatus(QStringLiteral("正在想故事"), QStringLiteral("小蝶听到了：") + heard.left(36), "#2563eb");
          appendLog(QStringLiteral("小蝶听到了：") + heard.left(60));
        }
      } else if (line.startsWith("[xiaodie_progress]")) {
        QString msg = line.section("] ", 1).trimmed();
        if (!msg.isEmpty()) {
          if (msg.contains(QStringLiteral("声音")) || msg.contains(QStringLiteral("播放"))) {
            setStatus(QStringLiteral("准备声音"), msg.left(40), "#0891b2");
          } else {
            setStatus(QStringLiteral("正在想故事"), msg.left(40), "#2563eb");
          }
          appendLog(msg);
        }
      } else if (line.startsWith("[xiaodie_played]")) {
        QString storyText = line.section("] ", 1).trimmed();
        if (!storyText.isEmpty()) {
          setStatus(QStringLiteral("讲故事中"), QStringLiteral("小蝶正在讲故事。"), "#0891b2");
          appendLog(storyText);
        }
      } else if (line.startsWith("[xiaodie_tts_done]")) {
        if (!storyFinishedShown_) {
          storyFinishedShown_ = true;
          setStatus(QStringLiteral("准备好了"), QStringLiteral("还想听新故事，可以继续按按钮。"), "#15935f");
          appendLog(QStringLiteral("故事讲完了，还可以继续点播新故事。"));
        }
      } else if (line.contains("story done")) {
        if (!storyFinishedShown_) {
          storyFinishedShown_ = true;
          setStatus(QStringLiteral("准备好了"), QStringLiteral("还想听新故事，可以继续按按钮。"), "#15935f");
          appendLog(QStringLiteral("故事讲完了，还可以继续点播新故事。"));
        }
      } else if (line.contains("ASR failed")) {
        setStatus(QStringLiteral("没听清"), QStringLiteral("可以再按住按钮说一遍。"), "#b7791f");
        appendLog(QStringLiteral("小蝶没有听清楚，可以再说一遍。"));
      }
    }
  }
};

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (QString::fromLocal8Bit(argv[i]) == "--check") {
      printf("xiaodie_qt_gui_ok\n");
      return 0;
    }
  }
  QApplication app(argc, argv);
  XiaoDieWindow window;
  window.show();
  return app.exec();
}
