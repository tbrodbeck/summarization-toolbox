''' create a Graphical User Interface (GUI) for summarizations '''
import os
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(os.getcwd())))
from modelTrainer import abstractive_summarizer
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QKeySequence, QWindow
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QSizePolicy, QGridLayout, QVBoxLayout, QPlainTextEdit, QPushButton, QTextEdit, QWidget

class UI(QMainWindow):
  ''' GUI '''
  def __init__(self, model: str):
      super().__init__()

      # create menu actions
      menubar = self.menuBar()
      windowMenu = menubar.addMenu('&Window')
      exitAction = QAction(' &Minimize', self)
      exitAction.setShortcut('Ctrl+M')
      exitAction.triggered.connect(self.minimize)
      windowMenu.addAction(exitAction)
      zoomInAction = QAction(' &Zoom in', self)
      zoomInAction.setShortcut(QKeySequence.ZoomIn)
      zoomInAction.triggered.connect(self.zoomIn)
      windowMenu.addAction(zoomInAction)
      zoomOutAction = QAction(' &Zoom out', self)
      zoomOutAction.setShortcut(QKeySequence.ZoomOut)
      zoomOutAction.triggered.connect(self.zoomOut)
      windowMenu.addAction(zoomOutAction)
      originalSizeAction = QAction(' &Original Size', self)
      originalSizeAction.setShortcut('Ctrl+0')
      originalSizeAction.triggered.connect(self.originalSize)
      windowMenu.addAction(originalSizeAction)

      self.summarizationLayout = QGridLayout()

      # label row
      self.summarizationLayout.addWidget(QLabel('Text'), 0, 0)
      self.summarizationLayout.addWidget(QLabel('T5 Summarizer'), 0, 1)

      # summarization rows
      self.input = QPlainTextEdit(placeholderText='Insert text to summarize')
      self.input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
      self.input.installEventFilter(self)  # press Enter to summarize
      self.summarizationLayout.addWidget(self.input, 1, 0)  # widget, position --> row index, column index
      self.summarizeButton = QPushButton('Summarize text')
      self.summarizeButton.clicked.connect(self.summarize)
      self.summarizationLayout.addWidget(self.summarizeButton, 1, 1)  # widget, position --> row index, column index

      mainLayout = QVBoxLayout()
      mainLayout.addLayout(self.summarizationLayout)
      uploadFileButton = QPushButton('Upload texts from a file')
      uploadFileButton.clicked.connect(self.uploadFile)
      mainLayout.addWidget(uploadFileButton)

      # setup central widget
      self.centralWidget = QWidget()
      self.setCentralWidget(self.centralWidget)
      self.centralWidget.setLayout(mainLayout)

      self.setWindowTitle('Sesame Street')

      self.summarizer = abstractive_summarizer.AbstractiveSummarizer(
                                                                    model,
                                                                    "german",
                                                                    status="fine-tuned"
                                                                )

  def eventFilter(self, obj, event):
    ''' activates self.summarize when Enter is pressed '''
    if event.type() == QtCore.QEvent.KeyPress and obj is self.input:
        if event.key() == QtCore.Qt.Key_Return and self.input.hasFocus():
            self.summarize()
            self.input.clear()
    return super().eventFilter(obj, event)

  def originalSize(self):
    ''' resets font size '''
    self.centralWidget.setFont(QFont(".AppleSystemUIFont", 13))

  def minimize(self):
    ''' minimize window in operating system '''
    self.setWindowState(self.windowState() | QWindow.Minimized)

  def summarize(self):
    ''' summarize the text in the input field '''
    textString = self.input.toPlainText()
    text = QTextEdit(textString)
    text.setReadOnly(True)
    self.input.clear()
    self.summarizationLayout.replaceWidget(self.input, text)
    summarization = self.summarizer.predict(textString)
    summary = QTextEdit(summarization)
    summary.setReadOnly(True)
    self.summarizationLayout.replaceWidget(self.summarizeButton, summary)
    nrRows = self.summarizationLayout.rowCount()
    self.summarizationLayout.addWidget(self.input, nrRows, 0)
    self.summarizationLayout.addWidget(self.summarizeButton, nrRows, 1)

  def uploadFile(self):
    print('todo')

  def zoomIn(self):
    ''' increases font size '''
    size = self.centralWidget.font().pointSize()
    self.centralWidget.setFont(QFont(".AppleSystemUIFont", size + 2))

  def zoomOut(self):
    ''' decreases font size '''
    size = self.centralWidget.font().pointSize()
    self.centralWidget.setFont(QFont(".AppleSystemUIFont", size - 2))

def run_gui(model: str):
  ''' execute the gui '''
  app = QApplication(sys.argv)
  ui = UI(model)
  ui.resize(800, 600)  # (width, length) default size in pixels
  ui.show()
  sys.exit(app.exec_())
