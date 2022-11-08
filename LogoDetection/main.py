import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import cv2
import matplotlib.pyplot as plt


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        #img = cv2.resize(image, (500, 500))
        #image = img
        super().setPixmap(image)

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setWindowTitle("Logo Detection")
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            show_result(file_path)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))

def show_result(text):
    msg = QMessageBox()
    msg.setWindowTitle("Result")
    msg.setText("Logo Test")
    x = msg.exec_()


app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())