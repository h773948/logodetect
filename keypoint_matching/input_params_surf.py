# importing libraries
from PyQt5.QtWidgets import *
from surf_detector import surf_detect
import sys

# creating a class
# that inherits the QDialog class
class Window(QDialog):

    # constructor
    def __init__(self):
        super(Window, self).__init__()

        # setting window title
        self.setWindowTitle("Surf Detector")

        # setting geometry to the window
        self.setGeometry(100, 100, 300, 400)

        # creating a group box
        self.formGroupBox = QGroupBox("Parameters")

        # creating spin box to select age
        self.intSpinBar = QSpinBox()
        self.doubleSpinBar = QDoubleSpinBox()
        self.doubleSpinBar.setSingleStep(0.1)
        # creating combo box to select degree
        self.degreeComboBox = QComboBox()

        self.mode_ransacCheckBox=QCheckBox()
        self.draw_bounding_boxCheckBox=QCheckBox()

        # adding items to the combo box
        self.degreeComboBox.addItems(["affine", "perspective"])

        # creating a line edit
        self.nameLineEdit = QLineEdit()

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
        print("lowe ratio : {0}".format(self.doubleSpinBar.value()))
        
        #print("Person Name : {0}".format(self.nameLineEdit.text()))
        print("searchmode : {0}".format(self.degreeComboBox.currentText()))
        print("Ransac : {0}".format(self.mode_ransacCheckBox.checkState()))
        print("Bounding box : {0}".format(self.draw_bounding_boxCheckBox.checkState()))
        
###############################################################################################################################
        """
        teszt.....
        """
        SRC_FOLDER = '../pic/'
        SOURCE_IMAGE1_NAME = 'ford.png'
        SOURCE_IMAGE2_NAME = 'ford2.jpg'
        # SOURCE_IMAGE2_NAME = '224886291.jpg'
        # SOURCE_IMAGE2_NAME = '255740214.jpg'
        SOURCE_IMAGE1 = SRC_FOLDER + SOURCE_IMAGE1_NAME
        SOURCE_IMAGE2 = SRC_FOLDER + SOURCE_IMAGE2_NAME
        surf_detect(self.doubleSpinBar.value(), self.degreeComboBox.currentText(), self.mode_ransacCheckBox.checkState(), self.draw_bounding_boxCheckBox.checkState(), SOURCE_IMAGE1, SOURCE_IMAGE2)
        
#################################################################################################################################
        

        
        
        # closing the window
        self.close()

    # creat form method
    def createForm(self):
        
        # creating a form layout
        layout = QFormLayout()
        
        # adding rows
        # for name and adding input text
        layout.addRow(QLabel("lowe ratio"), self.doubleSpinBar)
        
        #layout.addRow(QLabel("str"), self.nameLineEdit)
        

        # for degree and adding combo box
        layout.addRow(QLabel("search mode"), self.degreeComboBox)
        layout.addRow(QLabel("MODE_RANSAC"), self.mode_ransacCheckBox)
        layout.addRow(QLabel("DRAW_BOUNDING_BOX"), self.draw_bounding_boxCheckBox)

        # for age and adding spin box
        

        # setting layout
        self.formGroupBox.setLayout(layout)


# main method
if __name__ == '__main__':

    # create pyqt5 app
    app = QApplication(sys.argv)

    # create the instance of our Window
    window = Window()

    # showing the window
    window.show()

    # start the app
    sys.exit(app.exec())
