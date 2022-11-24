import sys
from input_params_orb import *
from input_params_surf import *
from input_params_cnn import *
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel, \
    QRadioButton, QFileDialog
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from cnn_detector import cnn_detector

logo = None
pic = None
surf_detector = False
orb_detector = False
cnn_detector = False
SOURCE_IMAGE1 = " "
SOURCE_IMAGE2 = " "
window1=None
window2=None

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Logo detection'
        self.left = 50
        self.top = 50
        self.width = 350
        self.height = 350
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1 = QLabel(self)
        self.label1.move(20, 20)
        self.label1.setText("Logo url:")
        self.label1.setFont(QFont('Arial', 12))

        self.but1 = QPushButton('Load logo', self)
        self.but1.move(150, 25)

        # Create textbox
        self.textbox1 = QLineEdit(self)
        self.textbox1.move(20, 60)
        self.textbox1.resize(280, 40)

        self.label2 = QLabel(self)
        self.label2.move(20, 100)
        self.label2.setText("Picture url:")
        self.label2.setFont(QFont('Arial', 12))

        self.but2 = QPushButton('Load picture', self)
        self.but2.move(150, 105)

        # Create textbox
        self.textbox2 = QLineEdit(self)
        self.textbox2.move(20, 140)
        self.textbox2.resize(280, 40)

        self.radiobutton1 = QRadioButton(self)
        self.radiobutton1.setText("surf_detector")
        self.radiobutton1.setGeometry(20, 175, 140, 60)
        self.radiobutton1.setChecked(True)
        self.radiobutton1.toggled.connect(self.surf_selected)
        #self.radioButton1.setGeometry(QtCore.QRect(180, 120, 95, 20))
        #self.radioButton1.toggled.connect(self.surf_selected)

        self.radiobutton2 = QRadioButton(self)
        self.radiobutton2.setText("orb_detector")
        self.radiobutton2.setGeometry(150, 175, 140, 60)
        self.radiobutton2.setChecked(False)
        self.radiobutton2.toggled.connect(self.orb_selected)
        #self.radioButton2.setGeometry(QtCore.QRect(180, 150, 95, 20))
        #self.radioButton2.toggled.connect(self.orb_selected)

        self.radiobutton3 = QRadioButton(self)
        self.radiobutton3.setText("cnn_detector")
        self.radiobutton3.setGeometry(280, 175, 140, 60)
        self.radiobutton3.setChecked(False)
        self.radiobutton3.toggled.connect(self.cnn_selected)
        #self.radioButton3.setGeometry(QtCore.QRect(180, 150, 95, 20))
        #self.radioButton3.toggled.connect(self.orb_selected)

        # Create a button in the window
        self.button = QPushButton('Next', self)
        self.button.move(20, 225)
        #self.button.accepted.connect(self.getInfo)
        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.but1.clicked.connect(self.browselogo)
        self.but2.clicked.connect(self.browsepicture)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox1.text()
        logo = self.textbox1.text()
        pic = self.textbox2.text()
        
        SOURCE_IMAGE1=logo
        SOURCE_IMAGE2=pic
        if(self.radiobutton2.isChecked()==True):
            print("logo ",SOURCE_IMAGE1)
            print("pic ",SOURCE_IMAGE2)
            #app1 = QApplication(sys.argv)
            self.window1= WindowO(logo,pic)
            self.window1.show()
            #sys.exit(app.exec())
            print("asd")
        elif(self.radiobutton1.isChecked()==True):
            print("logo ",SOURCE_IMAGE1)
            print("pic ",SOURCE_IMAGE2)
            #app1 = QApplication(sys.argv)
            self.window2 = WindowS(logo,pic)
            self.window2.show()
            #sys.exit(app.exec())
            print("asd1")
        elif(self.radiobutton3.isChecked()==True):
            print("logo ",SOURCE_IMAGE1)
            print("pic ",SOURCE_IMAGE2)
            #app1 = QApplication(sys.argv)
            self.window2 = WindowCNN(pic)
            self.window2.show()
            #sys.exit(app.exec())
            print("asd12")

        else:
            print("orb ",self.radiobutton1.isChecked())
            print("surf ",self.radiobutton2.isChecked())
            print("pic1 ",pic)



        #QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QMessageBox.Ok,
        #                     QMessageBox.Ok)
        #self.textbox1.setText("")

    def surf_selected(self, selected):
        if selected:
            #self.label.setText("You are male")
            surf_detector = True
            orb_detector = False
            cnn = False

    def orb_selected(self, selected):
        if selected:
            #self.label.setText("You are male")
            surf_detector = False
            cnn_detector = False
            orb_detector = True

    def cnn_selected(self, selected):
        if selected:
            #self.label.setText("You are male")
            surf_detector = False
            orb_detector = False
            cnn_detector = True

    def browselogo(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/ronya/Desktop/',"Image files (*.jpg *.png)")
        self.textbox1.setText(fname[0])

    def browsepicture(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/ronya/Desktop/',"Image files (*.jpg *.png)")
        self.textbox2.setText(fname[0])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())