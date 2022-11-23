import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel, \
    QRadioButton
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets

logo = None
pic = None
surf_detector = False
orb_detector = False

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

        # Create textbox
        self.textbox1 = QLineEdit(self)
        self.textbox1.move(20, 60)
        self.textbox1.resize(280, 40)

        self.label2 = QLabel(self)
        self.label2.move(20, 100)
        self.label2.setText("Picture url:")
        self.label2.setFont(QFont('Arial', 12))

        # Create textbox
        self.textbox2 = QLineEdit(self)
        self.textbox2.move(20, 140)
        self.textbox2.resize(280, 40)

        radiobutton1 = QRadioButton(self)
        radiobutton1.setText("surf_detector")
        radiobutton1.setGeometry(20, 175, 140, 60)
        radiobutton1.setChecked(True)
        radiobutton1.toggled.connect(self.surf_selected)
        #self.radioButton1.setGeometry(QtCore.QRect(180, 120, 95, 20))
        #self.radioButton1.toggled.connect(self.surf_selected)

        radiobutton2 = QRadioButton(self)
        radiobutton2.setText("orb_detector")
        radiobutton2.setGeometry(150, 175, 140, 60)
        radiobutton2.setChecked(False)
        radiobutton2.toggled.connect(self.orb_selected)
        #self.radioButton2.setGeometry(QtCore.QRect(180, 150, 95, 20))
        #self.radioButton2.toggled.connect(self.orb_selected)

        # Create a button in the window
        self.button = QPushButton('Next', self)
        self.button.move(20, 225)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox1.text()
        logo = self.textbox1.text()
        pic = self.textbox2.text()
        #QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QMessageBox.Ok,
        #                     QMessageBox.Ok)
        #self.textbox1.setText("")

    def surf_selected(self, selected):
        if selected:
            #self.label.setText("You are male")
            surf_detector = True
            orb_detector = False

    def orb_selected(self, selected):
        if selected:
            #self.label.setText("You are male")
            surf_detector = False
            orb_detector = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())