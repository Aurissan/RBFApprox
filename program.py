#main file

from PyQt5 import QtCore, QtGui, QtWidgets
import cx_Freeze
import newui
import sys
import input
import warnwindow
import predictwindow
import yesnodialog
import prederr
import numpy as np
from sklearn import preprocessing
import rbf
import time
from pathlib import Path
import pickle
import pandas as pd
import os

class WarningWindow(QtWidgets.QDialog, warnwindow.Ui_Dialog): #error window initialization
    def __init__(self, parent=None):
        super(WarningWindow, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(336, 220)
        self.pushButton.clicked.connect(self._buttonpress)
    def _buttonpress(self):
        self.close()

class PredictedErr(QtWidgets.QDialog, prederr.Ui_Dialog): #window for cho
    def __init__(self, parent=None):
        super(PredictedErr, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(336, 220)

class YesNo(QtWidgets.QDialog, yesnodialog.Ui_Dialog):
#window for choosing whether to search for file with output data to calculate the prediction error of network
    def __init__(self,parent=None):
        super(YesNo, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(336,220)
        self.label.setWordWrap(True)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Yes).setText("Да")
        self.buttonBox.button(QtWidgets.QDialogButtonBox.No).setText("Нет")

class PredictWindow(QtWidgets.QDialog, predictwindow.Ui_Dialog): #window with calculation results
    def __init__(self, parent=None):
        super(PredictWindow, self).__init__(parent)
        self.setupUi(self)
        self.warnwindow=WarningWindow()
        self.yesno=YesNo()
        self.prederr=PredictedErr()

        self.pushButton.setText("Сохранить результаты")
        self.pushButton_2.setText("Погрешность")

        self.res=[]
        self.header=[]
        #self.dir=str(Path(__file__).parent)
        if getattr(sys, 'frozen', False):
            # frozen
            self.dir = str(os.path.dirname(sys.executable))
        else:
            # unfrozen
            self.dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.NeuralNetwork=None
        self.workx=None
        self.worky=None
        #self.predy=None
        self.flag=False
        self.flog=False
        self.flagde=False
        self.scalerY=None

        self.pushButton.clicked.connect(self._savedata)
        self.pushButton_2.clicked.connect(self._predict)

    def getinfo(self, res, header, dir): #get first batch of parameters from main window
        self.res = res
        self.header=header
        self.dir=dir

    def getotherinfo(self, network, workx, worky, flagnorm, scaler, flagdenorm):
        # get second batch of parameters from main window
        self.NeuralNetwork=network
        self.workx=workx
        self.worky=worky
        self.flag=flagnorm
        self.scalerY=scaler
        self.flagde=flagdenorm

    def _rightdatacheck(self, obj):
    #check the file with output data used for error calculation

        if (not isinstance(obj, tuple)) and obj == "ER0":
            self.warnwindow.label.setText("Внимание!\n В файле отстутствуют "
                                       "названия столбцов. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER1":
            self.warnwindow.label.setText("Внимание!\n В файле неправильно указаны или отстутствует "
                                       "часть названий столбцов. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER2":
            self.warnwindow.label.setText(
                "Внимание!\n В файле встречаются нечисловые значения. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER3":
            self.warnwindow.label.setText(
                "Внимание!\n Выбран пустой файл. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER4":
            self.warnwindow.label.setText(
                "Внимание!\n Ошибка нормализации. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        elif not obj[0].tolist():
            self.warnwindow.label.setText(
                "Внимание!\n В файле нет значений. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warnwindow.show()
            return None
        else:
            return obj

    def _savedata(self): #save the outputs of network
        try:
            self.fullname = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить результат', self.dir,  self.tr("Excel files (*.xlsx);; Comma-separated values ( *.csv )"))
            self.filname=self.fullname[0]
            if self.filname:
                self.outframe=pd.DataFrame(self.res,columns=self.header)
                if self.fullname[1]=="Excel files (*.xlsx)":
                    self.outframe.to_excel(self.filname,index=False)
                elif self.fullname[1]=="Comma-separated values ( *.csv )":
                    self.outframe.to_csv(self.filname)
        except Exception as e: print(e)

    def _predict(self): #calculate the error of network prediction
        try:
            if  isinstance(self.worky, np.ndarray):
                self.err=self.NeuralNetwork.score(self.workx, self.worky)
                score = "Погрешность расчета: " + str(round(self.err * 100, 5)) + "%"

                self.prederr.label.setText(score)
                self.prederr.show()
            elif self.worky==None:
                self.yesno.label.setText("Отстуствуют данные для расчета погрешности. Желаете ли Вы загрузить их из файла?")
                t=self.yesno.exec_()
                if t:
                    self.predname=QtWidgets.QFileDialog.getOpenFileNames(self, 'Открыть файл', self.dir,  self.tr("Excel files (*.xlsx *.xls);; Comma-separated values ( *.csv );; Text files ( *.glo *.txt)"))[0]
                    if self.predname:
                        self.worky=input.readtrain(self.predname,predflag=True)
                        self.worky=self._rightdatacheck(self.worky)
                        if isinstance(self.worky, tuple) or isinstance(self.worky, np.ndarray):
                            if isinstance(self.worky, tuple):
                               self.worky=self.worky[1]
                            if self.worky.shape[1]==self.NeuralNetwork.w.shape[1] and self.worky.shape[0]==self.workx.shape[0]:

                                if self.flag and not self.flagde:
                                    self.worky=self.scalerY.transform(self.worky)
                                    self.flog=True
                                if self.flog and (self.flagde or not self.flag):
                                    self.worky = self.scalerY.inverse_transform(self.worky)
                                self.err = self.NeuralNetwork.score(self.workx, self.worky)
                                score = "Погрешность расчета: " + str(round(self.err * 100, 5)) + "%"

                                self.prederr.label.setText(score)
                                self.prederr.show()
                            else:
                                self.worky=None

                                self.warnwindow.label.setText(
                                    "Внимание!\n Размерность выбранных данных не совпадает с имеющимися. Проверьте правильность введенных данных и попробуйте еще раз")
                                self.warnwindow.show()

                else:
                    self.warnwindow.label.setText("Файл не был выбран.")
                    self.warnwindow.show()
        except Exception as e: print(e)


class RBFInterface(QtWidgets.QMainWindow, newui.Ui_MainWindow): #main window
    def __init__(self, parent=None):
        super(RBFInterface,self).__init__(parent)
        self.setupUi(self)

        self.warning=WarningWindow()
        self.predwindow=PredictWindow()

        self.label_image.setFixedSize(250,190)

        self.temptrainfnames=[]
        self.trainfnames=[]
        self.tempworkfnames=[]
        self.workfnames=[]
        self.tempcentfnames=[]
        self.centfnames=[]
        self.RBFnames = ['Гаусса','Лапласа','Коши','Кусочно-линейная','Вигнера']
        #self.dir=str(Path(__file__).parent)
        if getattr(sys, 'frozen', False):
            # frozen
            self.dir = str(os.path.dirname(sys.executable))
        else:
            # unfrozen
            self.dir = str(os.path.dirname(os.path.realpath(__file__)))

        #self.dir2 = str(Path(__file__).parent) + '/'
        if getattr(sys, 'frozen', False):
            # frozen
            self.dir2 = str(os.path.dirname(sys.executable)) + '/'
        else:
            # unfrozen
            self.dir2 = str(os.path.dirname(os.path.realpath(__file__))) + '/'
        self.centskip=1
        self.traindata=[]
        self.centers = []
        self.tempcents=[]
        self.tempcents2=[]
        self.feat_range=[]
        self.func=''
        self.NeuralNetwork=None
        self.scalerX=None
        self.scalerY=None
        self.workx=[]
        self.worky=None

        self.nocentloadflag=False

        self.radioButton_rand.setEnabled(False)
        self.radioButton.setEnabled(False)
        self.radioButton_2.setEnabled(False)
        self.label_10.setEnabled(False)
        self.label_11.setEnabled(False)
        self.checkBox_2.setEnabled(False)

        self.WorkPushButton.setEnabled(False)

        self.TrainDatPushButton.clicked.connect(self._gettrainfilenames)
        self.checkBox.stateChanged.connect(self._checkskip)
        self.TrainLoadButton.clicked.connect(self._trainload)
        self.TrainNormalizecheckBox.stateChanged.connect(self._trainnorm)

        self.WorkPushButton.clicked.connect(self._getworkfilenames)
        self.checkBox_3.stateChanged.connect(self._workskip)
        self.WorkLoadButton.clicked.connect(self._workload)

        if self.comboBoxFunc.currentText()==self.RBFnames[0]:
            self.func="gauss"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)
        self.comboBoxFunc.currentIndexChanged.connect(self._changeimage)

        self.radioButton_centrain.toggle()
        if self.radioButton_centrain.isChecked():
            self._centrainskip()
        self.radioButton_centrain.toggled.connect(self._centrainskip)
        self.radioButton_rand.toggled.connect(self._centrand)
        self.radioButton.toggled.connect(self._lincent)
        self.radioButton_2.toggled.connect(self._filecent)
        self.TrainDatPushButton_2.clicked.connect(self._choosecentfiles)
        self.TrainLoadButton_2.clicked.connect(self._centload)
        self.pushButton_fit.clicked.connect(self._fit)

        self.pushButton_fit.setEnabled(False)
        self.pushButton_predict.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.pushButton_saveweights.setEnabled(False)

        self.radioButton_3.toggle()

        self.pushButton.clicked.connect(self._tranerr)
        self.pushButton_saveweights.clicked.connect(self._save)
        self.pushButton_loadweights.clicked.connect(self._load)
        self.pushButton_predict.clicked.connect(self._predict)

    def _rightdatacheck(self, obj): #check the files for exceptions

        if (not isinstance(obj, tuple)) and obj == "ER0":
            self.warning.label.setText("Внимание!\n В файле отстутствуют "
                                       "названия столбцов. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER1":
            self.warning.label.setText("Внимание!\n В файле неправильно указаны или отстутствует "
                                       "часть названий столбцов. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER2":
            self.warning.label.setText("Внимание!\n В файле встречаются нечисловые значения. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER3":
            self.warning.label.setText(
                "Внимание!\n Выбран пустой файл. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        elif (not isinstance(obj, tuple)) and obj == "ER4":
            self.warning.label.setText(
                "Внимание!\n Ошибка нормализации. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        elif not obj[0].tolist():
            self.warning.label.setText(
                "Внимание!\n В файле нет значений. Проверьте правильность введенных данных и попробуйте еще раз.")
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")
            return None
        else: return obj

    def _changeimage(self): #change the image based on the selected RBF
        if self.comboBoxFunc.currentText()==self.RBFnames[0]:
            self.func = "gauss"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)
        elif self.comboBoxFunc.currentText()==self.RBFnames[1]:
            self.func = "laplace"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1-1.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)
        elif self.comboBoxFunc.currentText()==self.RBFnames[2]:
            self.func = "cauchy"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1-4.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)
        elif self.comboBoxFunc.currentText()==self.RBFnames[3]:
            self.func = "linear2"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1-5.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)
        else:
            self.func = "wigner"
            self.pixmap=QtGui.QPixmap(self.dir2+'img/Figure_1-6.png')
            self.label_image.setPixmap(self.pixmap)
            self.label_image.setScaledContents(True)

    def _changedir(self, fname): #remember the directory of the last file
        index=1
        while index < len(fname):
            if fname[-index] == '/':
                self.dir=fname[:-index+1]
                break
            index+=1

    def _gettrainfilenames(self): #get filename for training data file and change ui elements accordingly
        self.temptrainfnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Открыть файл', self.dir, self.tr("Excel files (*.xlsx *.xls);; Comma-separated values ( *.csv );; Text files ( *.glo *.txt)"))[0]
        if self.temptrainfnames:
            self.trainfnames=self.temptrainfnames
            if (len(self.trainfnames) == 1):
                self.listWidget.addItem("Выбран файл обучающей выборки:")
                self.listWidget.addItem(self.trainfnames[0])
            else:
                self.listWidget.addItem("Выбраны файлы обучающей выборки:")
                for i in self.trainfnames:
                    self.listWidget.addItem(i)
            self.listWidget.addItem("")
        if self.trainfnames:
            self.TrainLoadButton.setEnabled(True)
            self._changedir(self.trainfnames[0])
            self.TrainNormalizecheckBox.setEnabled(True)
            self.pushButton_fit.setEnabled(False)
            self.WorkPushButton.setEnabled(False)
            if self.TrainNormalizecheckBox.isChecked():
                self.comboBox.setEnabled(True)

    def _getworkfilenames(self): #get filename for file containing input data that we want to obtain outputs for (testing data)
        self.tempworkfnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Открыть файл', self.dir, self.tr("Excel files (*.xlsx *.xls);; Comma-separated values ( *.csv );; Text files ( *.glo *.txt)"))[0]
        if self.tempworkfnames:
            self.workfnames=self.tempworkfnames
            if (len(self.workfnames)==1):
                self.listWidget.addItem("Выбран файл для расчета:")
                self.listWidget.addItem(self.workfnames[0])
            else:
                self.listWidget.addItem("Выбраны файлы для расчета:")
                for i in self.workfnames:
                    self.listWidget.addItem(i)
            self.listWidget.addItem("")
        if self.workfnames:
            self.WorkLoadButton.setEnabled(True)
            self._changedir(self.workfnames[0])



    def _checkskip(self):
        if self.checkBox.isChecked():
            self.spinBox_dataskip.setEnabled(True)
        else: self.spinBox_dataskip.setEnabled(False)

    def _trainnorm(self):
        if self.TrainNormalizecheckBox.isChecked():
            self.comboBox.setEnabled(True)
        else: self.comboBox.setEnabled(False)

    def _trainload(self): #load and preprocess training data
        try:
            self.normrange = self.comboBox.currentText()
            self.templist = self.normrange.split(',')
            self.feat_range = []
            for i in self.templist:
                self.feat_range.append(int(i))
            self.templist = None
            self.trainskips = 1
            if self.checkBox.isChecked():
                self.trainskips=self.spinBox_dataskip.value()+1
            if self.TrainNormalizecheckBox.isChecked():
                self.normrange = self.comboBox.currentText()
                self.templist = self.normrange.split(',')
                self.feat_range = []
                for i in self.templist:
                    self.feat_range.append(int(i))
                self.templist = None
            try:
                #traindata=[]
                traindata = input.readtrain(self.trainfnames, self.trainskips)
                traindata = self._rightdatacheck(traindata)
                if isinstance(traindata, tuple) and np.array_equal(traindata[0],traindata[1]):
                    traindata=None
                    self.warning.label.setText("Внимание!\n В файле неправильно указаны или отстутствуют "
                                       "столбцы зависимой переменной. Проверьте правильность введенных данных и попробуйте еще раз.")
                    self.warning.show()
                    self.listWidget.addItem("Ошибка")
                    self.listWidget.addItem("")
            except Exception as e:
                self.warning.label.setText("Произошла неизвестная ошибка. Пожалуйста, попробуйте еще раз.")
                self.warning.show()
                self.listWidget.addItem("Ошибка")
                self.listWidget.addItem("")
            if traindata==None:
                self.TrainLoadButton.setEnabled(False)
                self.radioButton_rand.setEnabled(False)
                self.radioButton.setEnabled(False)
                self.radioButton_2.setEnabled(False)
                self.label_10.setEnabled(False)
                self.label_11.setEnabled(False)
            else:
                self.traindata=traindata

                self.trainx = self.traindata[0]
                self.trainy = self.traindata[1]
                if self.TrainNormalizecheckBox.isChecked():
                    self.scalerX=preprocessing.MinMaxScaler(feature_range=self.feat_range)
                    self.scalerY = preprocessing.MinMaxScaler(feature_range=self.feat_range)
                    self.trainx=self.scalerX.fit_transform(self.trainx)
                    self.trainy=self.scalerY.fit_transform(self.trainy)
                    self.checkBox_2.setEnabled(True)

                self.TrainNormalizecheckBox.setEnabled(False)
                self.comboBox.setEnabled(False)
                self.WorkPushButton.setEnabled(True)
                self.radioButton_rand.setEnabled(True)
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.label_10.setEnabled(True)
                self.label_11.setEnabled(True)
                self.pushButton_fit.setEnabled(True)
                self.traindata=traindata
                self.listWidget.addItem("Обучающая выборка успешно загружена")
                self.listWidget.addItem("")
        except Exception as e:
            self.warning.label.setText(str(e))
            self.warning.show()



    def _workskip(self):
        if self.checkBox_3.isChecked():
            self.spinBox.setEnabled(True)
        else: self.spinBox.setEnabled(False)

    def _workload(self): #load and preprocess testing data
        try:
            self.workskip=1
            if self.checkBox_3.isChecked():
                self.workskip=self.spinBox.value()+1
            try:
                self.tempwork = input.readtrain(self.workfnames,self.workskip)
                self.tempwork = self._rightdatacheck(self.tempwork)
                if self.tempwork!=None:
                    if self.NeuralNetwork:
                        if self.tempwork[0].shape[1]!=self.NeuralNetwork.centers.shape[1]:
                            self.tempwork=None
                            self.warning.label.setText(
                                "Внимание!\n Размерность рабочей выборки не совпадает с размерностью обучающей выборки!")
                            self.warning.show()
                            self.listWidget.addItem("Ошибка")
                            self.listWidget.addItem("")
                    elif self.traindata != []:
                        if self.tempwork[0].shape[1]!=self.trainx.shape[1]:
                            self.tempwork = None
                            self.warning.label.setText(
                                "Внимание!\n Размерность рабочей выборки не совпадает с размерностью обучающей выборки!")
                            self.warning.show()
                            self.listWidget.addItem("Ошибка")
                            self.listWidget.addItem("")
            except Exception as e:
                self.warning.label.setText(str(e))
                self.warning.show()
                self.listWidget.addItem("Ошибка")
                self.listWidget.addItem("")
                self.tempwork = None
            if self.tempwork == None:
                self.WorkLoadButton.setEnabled(False)
                self.pushButton_predict.setEnabled(False)
            else:
                self.workx=self.tempwork[0]
                self.worky=self.tempwork[1]
                if np.array_equal(self.workx,self.worky):
                    self.worky=None
                if self.TrainNormalizecheckBox.isChecked():
                    self.workx=self.scalerX.transform(self.workx)
                    if isinstance(self.worky, np.ndarray):
                        self.worky=self.scalerY.transform(self.worky)
                self.listWidget.addItem("Рабочая выборка успешно загружена")
                self.listWidget.addItem("")
                if self.NeuralNetwork: self.pushButton_predict.setEnabled(True)
        except Exception as e:
            self.warning.label.setText(str(e))
            self.warning.show()

    def _centrainskip(self):
        if self.nocentloadflag:
            self.pushButton_fit.setEnabled(True)
        if self.radioButton_centrain.isChecked():
            self.spinBox_centers.setEnabled(True)
        else:
            self.spinBox_centers.setEnabled(False)

    def _centrand(self):
        if self.nocentloadflag:
            self.pushButton_fit.setEnabled(True)
        if self.radioButton_rand.isChecked():
            self.spinBox_rand.setEnabled(True)
        else:
            self.spinBox_rand.setEnabled(False)

    def _lincent(self):
        if self.nocentloadflag:
            self.pushButton_fit.setEnabled(True)
        if self.radioButton.isChecked():
            self.spinBox_equalcen.setEnabled(True)
        else:
            self.spinBox_equalcen.setEnabled(False)

    def _filecent(self): #enable/disable some ui elements for choosing to load centers from a file
        try:
            if not isinstance(self.centers,np.ndarray) and (self.centers==None or self.centers==[]) :  #(isinstance(self.centers,np.ndarray) and self.centers.size==0)
                self.pushButton_fit.setEnabled(False)
                self.nocentloadflag=True
            elif isinstance(self.centers,np.ndarray) and self.centers.size==0:
                self.pushButton_fit.setEnabled(False)
                self.nocentloadflag = True
            if self.centfnames:
                self.TrainLoadButton_2.setEnabled(True)
            if self.radioButton_2.isChecked():
                self.TrainDatPushButton_2.setEnabled(True)
            else:
                self.TrainDatPushButton_2.setEnabled(False)
                self.TrainLoadButton_2.setEnabled(False)
        except Exception as e:
            self.warning.label.setText(str(e))
            self.warning.show()

    def _choosecentfiles(self): #get filename of a file with RBF centers

        self.tempcentfnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Открыть файл', self.dir, self.tr("Excel files (*.xlsx *.xls);; Comma-separated values ( *.csv );; Text files ( *.glo *.txt)"))[0]
        if self.tempcentfnames:
            self.centfnames=self.tempcentfnames
            if (len(self.centfnames)==1):
                self.listWidget.addItem("Выбран файл центров РБФ:")
                self.listWidget.addItem(self.centfnames[0])
            else:
                self.listWidget.addItem("Выбраны файлы центров РБФ:")
                for i in self.centfnames:
                    self.listWidget.addItem(i)
            self.listWidget.addItem("")
        if self.centfnames:
            self.TrainLoadButton_2.setEnabled(True)
            self._changedir(self.centfnames[0])

    def _centload(self): #load and preprocess centers from file

        try:
            try:
                self.tempcents=input.readtrain(self.centfnames)
                self.tempcents=self._rightdatacheck(self.tempcents)
                if self.tempcents!=None:
                    if self.tempcents[0].shape[1]!=self.trainx.shape[1]:
                        self.tempcents=None
                        self.warning.label.setText("Внимание!\n Размерность центров не совпадает с размерностью обучающей выборки!")
                        self.warning.show()
                        self.listWidget.addItem("Ошибка")
                        self.listWidget.addItem("")
            except Exception as e:
                self.warning.label.setText(str(e))
                self.warning.show()
                self.listWidget.addItem("Ошибка")
                self.listWidget.addItem("")
                self.tempcents=None
            if self.tempcents==None:
                self.TrainLoadButton_2.setEnabled(False)
            else:
                self.tempcents2=self.tempcents[0]

                if self.TrainNormalizecheckBox.isChecked():
                    self.tempcents2=self.scalerX.transform(self.tempcents2)
                self.listWidget.addItem("Центры РБФ успешно загружены")
                self.listWidget.addItem("")
                self.pushButton_fit.setEnabled(True)
        except Exception as e:
            self.warning.label.setText(str(e))
            self.warning.show()

    def _fit(self): #train the network
        try:
            if self.radioButton_centrain.isChecked():
                self.centskip=self.spinBox_centers.value()
                if self.spinBox_centers.value() == 0: self.centskip = 1
                self.centers = self.trainx[::self.centskip]
            elif self.radioButton_rand.isChecked():
                self.randnums=self.spinBox_rand.value()
                self.centers=input.rancen(self.trainx, self.randnums)
            elif self.radioButton.isChecked():
                self.numlin=self.spinBox_equalcen.value()
                if self.numlin==0: self.numlin = 1
                self.centers = input.lincen(self.trainx,self.numlin)
            elif self.radioButton_2.isChecked():
                if self.tempcents2!=[]:
                    self.centers=self.tempcents2
            #self.listWidget.addItem("Начало обучения")
            if self.radioButton_3.isChecked():
                self.NeuralNetwork = rbf.RBF(self.centers, float(self.doubleSpinBox.value()), self.func)
            else:
                try:
                    self.NeuralNetwork = rbf.TwoLayerRBF(self.centers, float(self.doubleSpinBox.value()), self.func)
                except Exception as e:
                    self.warning.label.setText(str(e))
                    self.warning.show()

            start=time.time()
            self.NeuralNetwork.fit(self.trainx, self.trainy)
            finish=time.time()
            self.listWidget.addItem("Обучение завершено")
            self.listWidget.addItem("Время обучения:")
            tm=round(finish-start,3)
            if tm > 60:
                tm/=60
                tm=str(tm) + "мин"
            else: tm = str(tm) + "c"
            self.listWidget.addItem(tm)
            self.listWidget.addItem("")
            self.pushButton_saveweights.setEnabled(True)
            self.pushButton.setEnabled(True)
            if self.workx!=[]:
                self.pushButton_predict.setEnabled(True)
        except Exception as e:
            self.warning.label.setText(str(e))
            self.warning.show()
            self.listWidget.addItem("Ошибка")
            self.listWidget.addItem("")

    def _tranerr(self): #calculate training error
        score=self.NeuralNetwork.score(self.trainx,self.trainy)
        self.listWidget.addItem("Погрешность обучения:")
        score = str(round(score*100,5)) + "%"
        self.listWidget.addItem(score)

    def _save(self): #pickle the NN model and some additional parameters
        self.savename=QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить нейронную сеть', self.dir, self.tr("RBFsave file (*.rbf)"))[0]
        if self.savename:
            self._changedir(self.savename)
            pickle_file=open(self.savename, "wb")
            pickleout=[self.NeuralNetwork, self.scalerX, self.scalerY]
            pickle.dump(pickleout,pickle_file)
            pickle_file.close()
            self.listWidget.addItem("Нейронная сеть успешно сохранена")
            self.listWidget.addItem("")
        else:
            self.listWidget.addItem("Нейронная сеть не была сохранена")
            self.listWidget.addItem("")

    def _load(self): #load the pickled model into program
        self.rbffile = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', self.dir, self.tr("RBFsave file (*.rbf)"))[0]
        if self.rbffile:
            self._changedir(self.rbffile)
            network_pickle=open(self.rbffile, "rb")
            self.NeuralNetwork, self.scalerX, self.scalerY =pickle.load(network_pickle)
            network_pickle.close()
            self.listWidget.addItem("Нейронная сеть успешно загружена")
            self.listWidget.addItem("")
            if self.scalerX!=None and self.scalerY!=None:
                if self.scalerX.feature_range == [-1,1]: self.comboBox.setCurrentIndex(1)
                else: self.comboBox.setCurrentIndex(0)
                self.TrainNormalizecheckBox.setChecked(True)
                self.TrainNormalizecheckBox.setEnabled(False)
                self.comboBox.setEnabled(False)
                self.checkBox_2.setEnabled(True)
            else:
                self.TrainNormalizecheckBox.setEnabled(False)
                self.comboBox.setEnabled(False)

            self.WorkPushButton.setEnabled(True)


    def _predict(self): #predict outputs based on input data and put the results in a new window
        try:
            self.listWidget.addItem("Начало расчета")

            start=time.time()
            #app.processEvents()
            #QtWidgets.QApplication.processEvents()
            self.predy = self.NeuralNetwork.predict(self.workx)

            finish=time.time()
            self.listWidget.addItem("Расчет успешно завершен")
            self.listWidget.addItem("Время расчета:")
            tm = round(finish - start, 3)
            if tm > 60:
                tm /= 60
                tm = str(tm) + "мин"
            else:
                tm = str(tm) + "c"
            self.listWidget.addItem(tm)
            self.listWidget.addItem("")

            if self.checkBox_2.isChecked():
                self.predy=self.scalerY.inverse_transform(self.predy)
                self.workx=self.scalerX.inverse_transform(self.workx)
            self.out=np.concatenate((self.workx,self.predy), axis=1)
            self.predwindow.tableWidget.setColumnCount(self.out.shape[1])
            self.predwindow.tableWidget.setRowCount(self.out.shape[0])
            self.header=[]
            for i in range (0, self.workx.shape[1]):
                self.header.append('x'+str(i+1))
            for i in range (0, self.predy.shape[1]):
                self.header.append('y'+str(i+1))

            self.predwindow.tableWidget.setHorizontalHeaderLabels(self.header)
            for i in range (0, self.out.shape[0]):
                for j in range (0, self.out.shape[1]):
                    self.predwindow.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(round(self.out[i][j],4))))
            self.predwindow.tableWidget.resizeColumnsToContents()

            if self.TrainNormalizecheckBox.isChecked() and self.checkBox_2.isChecked():
                self.workx=self.scalerX.transform(self.workx)
            self.predwindow.getinfo(self.out, self.header, self.dir)
            self.predwindow.getotherinfo(self.NeuralNetwork,self.workx,self.worky, self.TrainNormalizecheckBox.isChecked, self.scalerY, self.checkBox_2.isChecked())
            self.predwindow.show()
        except Exception as e: print (e)


app = QtWidgets.QApplication(sys.argv)
form = RBFInterface()
form.show()
app.exec_()