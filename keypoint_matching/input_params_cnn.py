# importing libraries
from PyQt5.QtWidgets import *
from cnn_detector import cnn_detector
import sys


# creating a class
# that inherits the QDialog class
class WindowCNN(QDialog):

    # constructor
    def __init__(self, pic):
        super(WindowCNN, self).__init__()
        self.pic=pic
        # setting window title
        self.setWindowTitle("CNN Detector")

        # setting geometry to the window
        self.setGeometry(100, 100, 300, 400)

        # creating a group box
        self.formGroupBox = QGroupBox("Parameters")

        # creating spin box to select age
        self.windowSizeInput = QSpinBox()
        self.windowSizeInput.setMaximum(10_000)
        self.strideInput = QSpinBox()
        self.strideInput.setMaximum(10_000)

        # calling the method that create the form
        self.createForm()

        # creating a dialog button for ok and cancel
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # adding action when form is accepted
        self.buttonBox.accepted.connect(self.getInfo)

        # adding action when form is rejected
        self.buttonBox.rejected.connect(self.reject)

        # creating a vertical layout
        mainLayout = QVBoxLayout()

        # adding form group box to the layout
        mainLayout.addWidget(self.formGroupBox)

        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)

        # setting lay out
        self.setLayout(mainLayout)

    # get info method called when form is accepted
    def getInfo(self):

        # printing the form information
        # print("Maxpoints : {0}".format(self.intSpinBar.text()))
        
        # #print("Person Name : {0}".format(self.nameLineEdit.text()))
        # print("searchmode : {0}".format(self.searchComboBox.currentText()))
        # print("Ransac : {0}".format(self.mode_ransacCheckBox.checkState()))
        # print("Bounding box : {0}".format(self.draw_bounding_boxCheckBox.checkState()))


###############################################################################################################################
        """
        teszt.....
        
        SRC_FOLDER = '../pic/'
        SOURCE_IMAGE1_NAME = 'ford.png'
        SOURCE_IMAGE2_NAME = 'ford2.jpg'
        # SOURCE_IMAGE2_NAME = '224886291.jpg'
        # SOURCE_IMAGE2_NAME = '255740214.jpg'
        SOURCE_IMAGE1 = SRC_FOLDER + SOURCE_IMAGE1_NAME
        SOURCE_IMAGE2 = SRC_FOLDER + SOURCE_IMAGE2_NAME
        """
        #orb_detect(self.intSpinBar.text(), self.searchComboBox.currentText(), self.mode_ransacCheckBox.checkState(), self.draw_bounding_boxCheckBox.checkState(), SOURCE_IMAGE1, SOURCE_IMAGE2)
        cnn_detector(self.pic, ((int(self.windowSizeInput.text()), int(self.strideInput.text()),),))
#################################################################################################################################
        # closing the window
        self.close()

    # creat form method
    def createForm(self):
        
        # creating a form layout
        layout = QFormLayout()
        
        # adding rows
        # for name and adding input text
        
        layout.addRow(QLabel("Sliding window size"), self.windowSizeInput)
        layout.addRow(QLabel("Sliding window stride"), self.strideInput)
        #layout.addRow(QLabel("str"), self.nameLineEdit)
        
        # setting layout
        self.formGroupBox.setLayout(layout)


# main method
if __name__ == '__main__':


    # create pyqt5 app
    
    app = QApplication(sys.argv)

    # create the instance of our Window
    window = WindowCNN()

    # showing the window
    window.show()

    # start the app
    sys.exit(app.exec())
