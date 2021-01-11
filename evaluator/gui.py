''' create a Graphical User Interface (GUI) for summarizations '''
import os
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(os.getcwd())))
from modelTrainer.abstractive_summarizer import AbstractiveSummarizer
from PyQt5.QtCore import pyqtSignal, Qt, QEvent, QTimer, QThread
from PyQt5.QtGui import QFont, QKeySequence, QWindow
from PyQt5.QtWidgets import QAction, QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QFileDialog, QGridLayout, QScrollArea, QVBoxLayout, QPlainTextEdit, QPushButton, QSizePolicy, QTextEdit, QWidget
import signal
from timelogging.timeLog import log

class UI(QMainWindow):
  ''' GUI '''
  def __init__(self, model_dir: str):
      super().__init__()
      self.setAcceptDrops(True)

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

      central_layout = QVBoxLayout()

      # # label row
      header_layout = QHBoxLayout()
      text_label = QLabel('Text')
      text_label.setAlignment(Qt.AlignCenter)
      header_layout.addWidget(text_label)
      summarizer_label = QLabel('T5 Summarizer')
      summarizer_label.setAlignment(Qt.AlignCenter)
      header_layout.addWidget(summarizer_label)
      header_layout.setContentsMargins(0, 0, 0, 0)
      # central_layout.addLayout(header_layout)
      self.header_frame = QFrame()
      self.header_frame.setLayout(header_layout)
      central_layout.addWidget(self.header_frame)
      self.header_frame.hide()

      # summarization rows
      self.summarizationLayout = QGridLayout()
      self.input = QPlainTextEdit(placeholderText='Insert text to summarize')
      self.input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
      self.input.installEventFilter(self)  # press Enter to summarize
      self.summarizationLayout.addWidget(self.input, 1, 0)  # widget, position --> row index, column index
      setup_layout = QVBoxLayout()
      self.summarize_button = QPushButton('Summarize text box')
      self.summarize_button.clicked.connect(self.summarize_input_field)
      setup_layout.addStretch(1)
      setup_layout.addWidget(self.summarize_button)
      uploadFileButton = QPushButton('Upload text file')
      uploadFileButton.clicked.connect(self.upload_file_event)
      setup_layout.addWidget(uploadFileButton)
      setup_layout.addStretch(1)
      setup_layout.setContentsMargins(0, 0, 0, 0)
      self.setup_widget = QWidget()
      self.setup_widget.setLayout(setup_layout)
      self.summarizationLayout.addWidget(self.setup_widget, 1, 1)  # widget, position --> row index, column index

      self.scroll = QScrollArea()
      self.scroll.setFrameStyle(QFrame.NoFrame)
      self.scroll.setWidgetResizable(True)
      self.central_frame = QFrame()
      self.central_frame.setLayout(self.summarizationLayout)
      self.summarizationLayout.setContentsMargins(0, 0, 0, 0)
      self.scroll.setWidget(self.central_frame)

      central_layout.addWidget(self.scroll)
      self.central_widget = QWidget()
      self.central_widget.setLayout(central_layout)

      self.setCentralWidget(self.central_widget)
      self.setWindowTitle('Sesame Street')

      if model_dir != 'dev':
        self.summarizer = AbstractiveSummarizer(
                                                                      model_dir,
                                                                      "german",
                                                                      status="fine-tuned"
                                                                  )
      self.next_summary_position = 1

  def eventFilter(self, obj, event):
    ''' activates self.summarize when Enter is pressed '''
    if event.type() == QEvent.KeyPress and obj is self.input:
      if event.key() == Qt.Key_Return and self.input.hasFocus():
        self.summarize_input_field()
        return True
    return False

  def originalSize(self):
    ''' resets font size '''
    self.central_widget.setFont(QFont(".AppleSystemUIFont", 13))

  def minimize(self):
    ''' minimize window in operating system '''
    self.setWindowState(self.windowState() | QWindow.Minimized)

  def summarize_input_field(self):
    text = self.read_input_field()
    if text:
      self.summarize_string(text, self.next_summary_position)
      self.next_summary_position += 1
      self.append_new_row_to_layout()

  def read_input_field(self):
    textString = self.input.toPlainText()
    return textString.strip()

  def summarize_string(self, text: str, position_in_layout: int):
    ''' summarize the text in the input field '''
    text_input_field = QTextEdit(text)
    text_input_field.setReadOnly(True)
    self.summarizationLayout.addWidget(text_input_field, position_in_layout, 0)
    summarize_label = QLabel('Summarizing...')
    summarize_label.setAlignment(Qt.AlignCenter)
    self.summarizationLayout.addWidget(summarize_label, position_in_layout, 1)
    modelRunner = ModelRunner(self.summarizer, text, position_in_layout, parent=self)
    modelRunner.start()
    modelRunner.summary_output.connect(self.summarize_finished)
    self.input.setFocus()

  def summarize_finished(self, summary_output):
    self.header_frame.show()
    summary_text_field = QTextEdit(summary_output['summary'])
    summary_text_field.setReadOnly(True)
    position_in_layout = summary_output['position_in_layout']
    self.summarizationLayout.addWidget(summary_text_field, position_in_layout, 1)
    self.summarizationLayout.setRowMinimumHeight(position_in_layout, 120)

  def append_new_row_to_layout(self):
    nrRows = self.next_summary_position + 1
    self.summarizationLayout.addWidget(self.input, nrRows, 0)
    self.input.clear()
    self.summarizationLayout.addWidget(self.setup_widget, nrRows, 1)

  def upload_file_event(self):
    fileProps = QFileDialog.getOpenFileName(self, 'Open File')
    filename = fileProps[0]
    if filename:
      self.summarize_file(filename)

  def read_lines_from_file_path(self, file_path):
    with open(file_path, 'r') as f:
      texts = f.read()
    return texts.split('\n')

  def summarize_file(self, file_path):
    texts = self.read_lines_from_file_path(file_path)
    for i, text in enumerate(texts):
      self.summarize_string(text, self.next_summary_position + i)
      if i == 10:
        break
    self.next_summary_position += len(texts)
    self.append_new_row_to_layout()

  def dragEnterEvent(self, event):
      if event.mimeData().hasUrls():
          event.accept()
      else:
          event.ignore()

  def dropEvent(self, event):
      files = [u.toLocalFile() for u in event.mimeData().urls()]
      for f in files:
          self.summarize_file(f)

  def zoomIn(self):
    ''' increases font size '''
    size = self.central_widget.font().pointSize()
    self.central_widget.setFont(QFont(".AppleSystemUIFont", size + 2))

  def zoomOut(self):
    ''' decreases font size '''
    size = self.central_widget.font().pointSize()
    self.central_widget.setFont(QFont(".AppleSystemUIFont", size - 2))

class ModelRunner(QThread):
  def __init__(self, model: AbstractiveSummarizer, text: str, position_in_layout: int, parent: QMainWindow):
    super().__init__(parent=parent)
    self.model = model
    self.text = text
    self.position_in_layout = position_in_layout

  summary_output = pyqtSignal(dict)

  def run(self):
    summary = self.model.predict(self.text)
    self.summary_output.emit({"summary": summary, "position_in_layout": self.position_in_layout})

def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    QApplication.quit()

def run_gui(model_dir: str):
  """execute the gui
  Args:
      model_dir (str): directory of the summarization model to use
  """
  app = QApplication(sys.argv)
  ui = UI(model_dir)
  ui.resize(800, 600)  # (width, length) default size in pixels
  ui.show()
  start_with_control_c_support(app)

def start_with_control_c_support(app: QApplication):
  signal.signal(signal.SIGINT, sigint_handler)
  timer = QTimer()
  timer.timeout.connect(lambda: None)
  timer.start(100)
  sys.exit(app.exec_())

if __name__ == '__main__':
  run_gui('dev')
