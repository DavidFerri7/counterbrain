import sys

from PySide6 import QtWidgets, QtCore

import pyqtgraph as pg

from mne.preprocessing import ICA
from mne.filter import filter_data

from scipy.fftpack import fft,fftfreq

import json

from ui_app_eeg import Ui_MainWindow
import pyqtgraph as pg

from PySide6.QtWidgets import QTableWidgetItem,QGraphicsView, QPlainTextEdit, QApplication, QMainWindow, QComboBox , QLabel, QCheckBox, QMessageBox, QRadioButton, QListWidget, QLineEdit , QFileDialog
from PySide6.QtCore import Qt

from pathlib import Path

#from matplotlib.figure import Figure

import pyqtgraph.exporters
import jinja2
import pdfkit
import os

#para acceder al editor de la app: copiar en "buscar" este enlace C:\Users\David\AppData\Local\Programs\Python\Python39\Lib\site-packages\PySide6

import sys
import numpy as np
import matplotlib.pyplot as plt
import mne.io as io_eeg
import mne 
#from mne import preprocessing
#from mne import viz
#from mne import decoding
import pandas as pd

import pywt.data

#from astroML.fourier import\
    #FT_continuous, IFT_continuous, sinegauss, sinegauss_FT, wavelet_PSD

#from mne import channels
#from mne.datasets import sample
from mne.channels import make_standard_montage
import scipy.signal as ss
#import scipy as sc
from scipy import signal
#import pyedflib

from spkit.data import load_data

from mne import viz
from mne import decoding
from mne import preprocessing

#import spkit as sp

from scipy.stats import linregress


def absPath(file):
    # Devuelve la ruta absoluta a un fichero desde el propio script
    return str(Path(__file__).parent.absolute() / file)



def get_relative_power(senal, f_low, f_high,fs ):
    fs = fs

    #f, Px = ss.welch(senal, fs=fs, window='hamming', nperseg=512, nfft=1024)
    f, Px  = ss.welch(senal, fs=fs )
    
    width = np.diff(f)
    
    #cojo todas las alturas menos la última, para que no chille por las dimensiones.
    pvec_total = np.multiply(width,Px[0:-1])
    avgp_total = np.sum(pvec_total);   
    
    #Puntos del vector de frecuencias que se corresponden con las frecuencias de corte
    f_low_index = (np.abs(f-f_low)).argmin()    
    f_high_index = (np.abs(f-f_high)).argmin()  

    #Calculo la integral entre esas dos frecuncias, se donde empieza y donde acaba.
    pvec = np.multiply(width[f_low_index:f_high_index], Px[f_low_index:f_high_index])
    avgp = np.sum(pvec);     # POTENCIA EN LA BANDA ELEGIDA
    
    #Ratio
    r_seg = avgp/avgp_total;  
    
    return [r_seg,f]



    
def get_ratios(senal, fs, band_1, band_2):
    fs = fs

    if band_1 == "delta":
        f_low, f_high = 1.5, 3.5
    elif band_1 == "theta":
        f_low, f_high = 3.5, 7.5
    elif band_1 == "alpha":
        f_low, f_high = 7.5, 12.5
    elif band_1 == "beta":
        f_low, f_high = 12.5, 30
    elif band_1 == "gamma":
        f_low, f_high = 30, 60
    
    if band_2 == "delta":
        f_low_2, f_high_2 = 1.5, 3.5
    elif band_2 == "theta":
        f_low_2, f_high_2 = 3.5, 7.5
    elif band_2 == "alpha":
        f_low_2, f_high_2 = 7.5, 12.5
    elif band_2 == "beta":
        f_low_2, f_high_2 = 12.5, 30
    elif band_2 == "gamma":
        f_low_2, f_high_2 = 30, 60


    #f, Px = ss.welch(senal, fs=fs, window='hamming', nperseg=512, nfft=1024)
    f, Px  = ss.welch(senal, fs=fs )
    
    width = np.diff(f)

    #Puntos del vector de frecuencias que se corresponden con las frecuencias de corte con la banda elegida 1
    f_low_index_1 = (np.abs(f-f_low)).argmin()    
    f_high_index_1 = (np.abs(f-f_high)).argmin()  

    #Puntos del vector de frecuencias que se corresponden con las frecuencias de corte con la banda elegida 2
    f_low_index_2 = (np.abs(f-f_low_2)).argmin()    
    f_high_index_2 = (np.abs(f-f_high_2)).argmin()  


    # Calculo la integral de la banda 1
    pvec_1 = np.multiply(width[f_low_index_1:f_high_index_1], Px[f_low_index_1:f_high_index_1])
    ratio_1 = np.sum(pvec_1);     # POTENCIA EN LA BANDA ELEGIDA
    
    # Calculo la integral de la banda 2
    pvec_2 = np.multiply(width[f_low_index_2:f_high_index_2], Px[f_low_index_2:f_high_index_2])
    ratio_2 = np.sum(pvec_2);   
    

    #Ratio
    r_seg = ratio_1/ratio_2;  
    
    return r_seg



class MainWindow(QMainWindow,Ui_MainWindow):

    #MODO NORMAL
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        self.grafica_eeg.setBackground("w")
        self.grafica_pendiente.setBackground("w")
        self.grafica_tf.setBackground("w")
        self.grafica_psd.setBackground("w")
        self.grafica_powers.setBackground("w")
        self.grafica_spectgram.setBackground("w")

        self.grafica_eeg.showGrid(x=True,y=True)
        self.grafica_pendiente.showGrid(x=True,y=True)
        self.grafica_tf.showGrid(x=True,y=True)
        self.grafica_psd.showGrid(x=True,y=True)
  

        self.actionCargar_edf.triggered.connect(self.abrir)

        self.show_ica_button.clicked.connect(self.show_ica)

        self.filt_ica_button.clicked.connect(self.filter_ica)

        self.actionSalir.triggered.connect(self.close_app)

        self.exportar_pdf.clicked.connect(self.generar_pdf)

        self.representar_signal.clicked.connect(self.plot_signal)

        self.Limpiar_signals.clicked.connect(self.clean_signal)

        self.show_power_button.clicked.connect(self.show_powers)
        
        self.representar_tf.clicked.connect(self.show_tf)

        self.representar_psd.clicked.connect(self.show_psd)

        self.representar_spectgram.clicked.connect(self.show_spectgram)

        self.plot_band.clicked.connect(self.plot_limits)

        self.plot_pendiente.clicked.connect(self.plot_pendiente_signal)

        self.plot_PSDWav.clicked.connect(self.show_PSDWav)

        self.plot_artefact.clicked.connect(self.plot_lines_artefact)

        self.show_power_ratios.clicked.connect(self.show_ratios)

        self.titulo_plot = []

    
    

#--------------------------------METODOS------------------------------------------

    def abrir(self):
        try:
            archivo = QFileDialog.getOpenFileName(self,"Abrir archivo","C:\\")
            self.file_name = absPath(archivo[0])  # str(os.path.basename(os.path.normpath(archivo[0])))
            self.eeg_object_init_1 = io_eeg.read_raw_edf(self.file_name,preload = True)
            self.t = self.eeg_object_init_1.times
            self.fs = self.eeg_object_init_1.info['sfreq']

            # cogemos sus nombres con .ch_names
            self.signal_labels = self.eeg_object_init_1.ch_names

            #para seleccionar la señal
            self.desplegable.clear()
            self.desplegable.addItems(self.signal_labels)   

            self.number = 1

            # creamos el desplegable de wavelets
            wavlist = pywt.wavelist(kind='continuous')
            wavlist = wavlist[10:20]
            self.desplegable_wavelets.addItems(wavlist)
            self.wav_select = self.desplegable_wavelets.currentText()

            # creamos el desplegable de bandas
            bands = ["delta","theta","alpha","beta","gamma"]
            self.ratio_1.addItems(bands)   
            self.ratio_2.addItems(bands)  

        
        except NotImplementedError:
            QtWidgets.QMessageBox.critical(self, "Ups.", f"Error leyendo el archivo, asegurese de que se trata de un arhivo .edf")


    
    def close_app(self):
        sys.exit()


    def show_ratios(self): 

        band_selected_1 = str(self.ratio_1.currentText())
        band_selected_2 = str(self.ratio_2.currentText())

        #potencias en banda
        self.ratio_selected = round(get_ratios(self.eeg[self.x,self.low:self.high], self.fs, band_selected_1, band_selected_2),3)
        
        self.show_ratio_text.setPlainText(str(self.ratio_selected))



    def plot_lines_artefact(self):

        self.art_low = int(self.artefact_hours_low.text())*3600 + int(self.artefact_minutes_low.text())*60 + int(self.artefact_seconds_low.text()) 
        #art_low = np.where(self.t == art_low)
        #self.art_low = art_low[0][0]

        self.art_high = int(self.artefact_hours_high.text())*3600 + int(self.artefact_minutes_high.text())*60 + int(self.artefact_seconds_high.text()) 
        #art_high = np.where(self.t == art_high)
        #self.art_high = art_high[0][0]


        vertical_line1 = pg.InfiniteLine(pos = self.art_low , angle=90, movable=True,  hoverPen= "black" ) #label=None, por si queremos indicar de k tipo es el artefacto
        vertical_line2 = pg.InfiniteLine(pos = self.art_high , angle=90, movable=True,  hoverPen= "black" ) #label=None, por si queremos indicar de k tipo es el artefacto

        vertical_line1.addMarker ( marker ="o" ,  size = 13.0)
        vertical_line2.addMarker ( marker ="o" ,  size = 13.0) 

        self.grafica_eeg.addItem(vertical_line1)
        self.grafica_eeg.addItem(vertical_line2)

    
        

    def plot_limits(self):
        #para representar una banda para plotear la pendiente
        self.low_banda = int(self.band_hours_low.text())*3600 + int(self.band_minutes_low.text())*60 + int(self.band_seconds_low.text()) 
        self.high_banda = int(self.band_hours_high.text())*3600 + int(self.band_minutes_high.text())*60 + int(self.band_seconds_high.text()) 
        self.rgn = pg.LinearRegionItem([self.low_banda, self.high_banda])  #donde colocarlo en eje x y eje y
        self.grafica_eeg.addItem(self.rgn)

        self.rgn.sigRegionChanged.connect(self.plot_pendiente_signal)
        
   


    def plot_pendiente_signal(self):
        self.grafica_pendiente.clear()
        self.grafica_pendiente.setBackground("w")
        self.grafica_pendiente.showGrid(x=True,y=True)

        self.region = self.rgn.getRegion()
        self.region_low = int(self.region[0])
        self.region_high = int(self.region[1]) + 1

        slope_low = int(self.slope_band_low.text())
        slope_high = int(self.slope_band_high.text()) + 1


        if len(self.titulo_plot) == 2:

            #PENDIENTE SEÑAL 1    
            x1 = self.signal_labels.index(self.titulo_plot[0])

            #spec = plt.specgram(self.eeg[x1,:] ,Fs=self.fs, cmap="prism")
            nperseg = self.fs
            ftf, ftt, ftZ = signal.stft(self.eeg[x1,:] , nperseg=nperseg, fs=self.fs)
            media = []
            for i in ftZ.T:
                #media.append(i.where(i.max()))
                media.append(np.where(i[slope_low:slope_high] == i[slope_low:slope_high].max())[0][0])   #i[8:14] para que mire solo en la banda alpha (8-13 Hz)
                #print(np.where(i == i.max())[0][0])

            time = ftt[self.region_low:self.region_high]  #TIEMPO
            y = media[self.region_low:self.region_high]
            #print(time)
            #print(y)

            coef = np.polyfit(time,y,1)
            poly1d_fn = np.poly1d(coef) 

           
            self.grafica_pendiente.plot(time*2, poly1d_fn(time*2), pen= pg.mkPen("blue",width = 1))


            #PENDIENTE SEÑAL 2  
            x2 = self.signal_labels.index(self.titulo_plot[1])
            #spec = plt.specgram(self.eeg[x2,:] ,Fs=self.fs, cmap="prism")
            nperseg = self.fs
            ftf, ftt, ftZ = signal.stft(self.eeg[x2,:] , nperseg=nperseg, fs=self.fs)
            media = []
            for i in ftZ.T:
                #media.append(i.where(i.max()))
                media.append(np.where(i[slope_low:slope_high] == i[slope_low:slope_high].max())[0][0])   #i[8:14] para que mire solo en la banda alpha (8-13 Hz)
                #print(np.where(i == i.max())[0][0])

            time = ftt[self.region_low:self.region_high]  #TIEMPO
            y = media[self.region_low:self.region_high]
            #print(time)
            #print(y)

            coef2 = np.polyfit(time,y,1)
            poly1d_fn = np.poly1d(coef2) 

            #self.grafica_pendiente.setTitle(f"Slope: {round(coef[0],1)}")   #(f"{self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "20px")
            self.grafica_pendiente.plot(time*2, poly1d_fn(time*2), pen= pg.mkPen("orange",width = 1))

            self.grafica_pendiente.setTitle(f"Slope - {self.titulo_plot[0]}:{round(coef[0],1)} | {self.titulo_plot[1]}:{round(coef2[0],1)}")   #(f"{self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "20px")
            self.grafica_pendiente.addLegend()


        if len(self.titulo_plot) == 1:
            #spec = plt.specgram(self.eeg[self.x,:] ,Fs=self.fs, cmap="prism")
            nperseg = self.fs
            ftf, ftt, ftZ = signal.stft(self.eeg[self.x,:] , nperseg=nperseg, fs=self.fs)
            media = []

            for i in ftZ.T:
                #media.append(i.where(i.max()))
                media.append(np.where(i[slope_low:slope_high] == i[slope_low:slope_high].max())[0][0])   #i[8:14] para que mire solo en la banda alpha (8-13 Hz)
                #print(np.where(i == i.max())[0][0])

            time = ftt[self.region_low:self.region_high]  #TIEMPO
            y = media[self.region_low:self.region_high]
            #print(time)
            #print(y)

            coef = np.polyfit(time,y,1)
            poly1d_fn = np.poly1d(coef) 

            self.grafica_pendiente.setTitle(f"Slope - {self.titulo_plot[0]}:{round(coef[0],1)}")   #(f"{self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "20px")
            self.grafica_pendiente.plot(time*2, poly1d_fn(time*2), pen= pg.mkPen("blue",width = 1))

            self.grafica_pendiente.addLegend()


        else:
            pass



    def plot_signal(self):

        try:   # sin filtro ICA
            condition = 1/self.number   #Provocamos un error si se le ha aplicado el filtro ICA para que no vuelva a coger los datos del archivo original
            #volvemos a leer el archivo para actualizar cada vez por si se cambian las bandas de filtrado
            self.eeg_object = self.eeg_object_init_1.copy()

            # FILTRAMOS CON NOTCH POR DEFECTO Y CON PASA BANDA ENTRE 1 Y 80 MODIFICABLE
            self.LB = float(self.low_band_filter.text())
            self.HB = float(self.high_band_filter.text())

            self.eeg_object.load_data().filter(l_freq= self.LB , h_freq= self.HB)
            self.eeg_object.load_data().notch_filter(freqs = 50)

            # leemos las señales ya filtradas
            self.eeg = self.eeg_object.get_data()
            
        except Exception:             
            
            self.eeg = self.eeg_ica_final   # si hemos pasado el filtro de ICA
            

        # nos quedamos con el valor del nombre de la señal que hay en el desplegable
        self.s = self.desplegable.currentText()
        self.x = self.signal_labels.index(self.s)  

        self.titulo_plot.append(self.s)
        if len(self.titulo_plot) > 2:
            self.titulo_plot.pop(2)

        low = int(self.button_hours_low.text())*3600  +  int(self.button_minutes_low.text())*60  +  int(self.button_seconds_low.text())
        low = np.where(self.t == low)
        self.low = low[0][0]

        high = int(self.button_hours_high.text())*3600  +  int(self.button_minutes_high.text())*60  +  int(self.button_seconds_high.text()) 
        high = np.where(self.t == high)
        self.high = high[0][0]
        
        self.grafica_eeg.addLegend()
        self.grafica_eeg.setBackground("w")


        if len(self.titulo_plot) == 2:
            self.grafica_eeg.clear()

            #PARA REPRESENTAR EEG
            self.grafica_eeg.setTitle(f"{self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "20px")
            x1 = self.signal_labels.index(self.titulo_plot[0])
            self.grafica_eeg.plot(self.t[self.low:self.high],self.eeg[x1,self.low:self.high],pen= pg.mkPen("blue",width = 1), name= f"{self.titulo_plot[0]}")

            x2 = self.signal_labels.index(self.titulo_plot[1])
            self.grafica_eeg.plot(self.t[self.low:self.high],self.eeg[x2,self.low:self.high],pen= pg.mkPen("orange",width = 1), name= f"{self.titulo_plot[1]}")

            self.grafica_eeg.showGrid(x=True,y=True)

    
        
        if len(self.titulo_plot) == 1:
            #PARA REPRESENTAR EEG
            self.grafica_eeg.clear()
            self.grafica_eeg.setTitle(f"{self.s}", size = "20px")
            self.grafica_eeg.plot(self.t[self.low:self.high],self.eeg[self.x,self.low:self.high],pen= pg.mkPen("blue",width = 1), name= f"{self.s}")
            self.grafica_eeg.showGrid(x=True,y=True)

        else:
            pass

    

        styles = {"color":"#000", "font-size":"13px"}
        self.grafica_eeg.setLabel("left",'Amplitude [uV]',**styles)
        self.grafica_eeg.setLabel("bottom",'time [s]',**styles)



    def clean_signal(self):
        self.titulo_plot = []
        self.grafica_eeg.clear()
        #self.desplegable.clear()
        self.grafica_tf.clear()
        self.tableWidget.clear()
        self.grafica_psd.clear()
        self.grafica_pendiente.clear()
        self.grafica_powers.clear()
        self.grafica_spectgram.clear()
        self.show_ratio_text.setPlainText("None")
        self.textEdit.clear()


    def show_powers(self):
        self.tableWidget.clear()

        # para que no se pueda modificar los datos de la tabla
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)


        styles = {"color":"#000", "font-size":"13px"}
        
        #potencias en banda
        self.r_seg_delta = round(get_relative_power(self.eeg[self.x,self.low:self.high],1.5,3.5,self.fs)[0],3)
        self.r_seg_theta = round(get_relative_power(self.eeg[self.x,self.low:self.high],3.5,7.5,self.fs)[0],3)
        self.r_seg_alpha = round(get_relative_power(self.eeg[self.x,self.low:self.high],7.5,12.5,self.fs)[0],3)
        self.r_seg_beta = round(get_relative_power(self.eeg[self.x,self.low:self.high],12.5,30,self.fs)[0],3)
        self.r_seg_gamma = round(get_relative_power(self.eeg[self.x,self.low:self.high],30,100,self.fs)[0],3)


        if len(self.titulo_plot) == 2:
            self.tableWidget.clear()
            self.grafica_powers.clear()
            self.grafica_eeg.setTitle(f"{self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "15px")
            

            #potencias en banda
            self.pos1 = self.signal_labels.index(str(self.titulo_plot[0])) 
            self.r_seg_delta = round(get_relative_power(self.eeg[self.pos1,self.low:self.high],1.5,3.5,self.fs)[0],3)
            self.r_seg_theta = round(get_relative_power(self.eeg[self.pos1,self.low:self.high],3.5,7.5,self.fs)[0],3)
            self.r_seg_alpha = round(get_relative_power(self.eeg[self.pos1,self.low:self.high],7.5,12.5,self.fs)[0],3)
            self.r_seg_beta = round(get_relative_power(self.eeg[self.pos1,self.low:self.high],12.5,30,self.fs)[0],3)
            self.r_seg_gamma = round(get_relative_power(self.eeg[self.pos1,self.low:self.high],30,100,self.fs)[0],3)

            self.pos2 = self.signal_labels.index(str(self.titulo_plot[1])) 
            self.r_seg_delta_2 = round(get_relative_power(self.eeg[self.pos2,self.low:self.high],1.5,3.5,self.fs)[0],3)
            self.r_seg_theta_2 = round(get_relative_power(self.eeg[self.pos2,self.low:self.high],3.5,7.5,self.fs)[0],3)
            self.r_seg_alpha_2 = round(get_relative_power(self.eeg[self.pos2,self.low:self.high],7.5,12.5,self.fs)[0],3)
            self.r_seg_beta_2 = round(get_relative_power(self.eeg[self.pos2,self.low:self.high],12.5,30,self.fs)[0],3)
            self.r_seg_gamma_2 = round(get_relative_power(self.eeg[self.pos2,self.low:self.high],30,100,self.fs)[0],3)

            self.nombres_filas = ["Ratio/Power delta", "Ratio/Power theta",
            "Ratio/Power alpha","Ratio/Power beta","Ratio/Power gamma"]
            
            # Creamos la estructura de datosX
            datos = []

            # Añadimos un registro manualmente
            datos.append({
                f"{self.titulo_plot[0]}": round(self.r_seg_delta,3),
                f"{self.titulo_plot[1]}": round(self.r_seg_delta_2,3),
            })

            # Definimos una lista de contactos
            contactos = [
                (round(self.r_seg_theta,3),round(self.r_seg_theta_2,3)),
                (round(self.r_seg_alpha,3),round(self.r_seg_alpha_2,3)),
                (round(self.r_seg_beta,3),round(self.r_seg_beta_2,3)),
                (round(self.r_seg_gamma,3),round(self.r_seg_gamma_2,3))
            ]

            # Añadimos un contactos dinámicamente
            for valor1,valor2 in contactos:
                datos.append({
                    f"{self.titulo_plot[0]}": valor1,
                    f"{self.titulo_plot[1]}": valor2
                })

            # Guardamos los registros del fichero
            with open(absPath("contactos.json"), "w") as fichero:
                json.dump(datos, fichero)

            # Leemos los registros del fichero
            with open(absPath("contactos.json")) as fichero:
                datos = json.load(fichero)
                

            # cargamos el contenido del fichero
            with open(absPath("contactos.json")) as fichero:
                self.datos = json.load(fichero)

            # definimos la configuración de las columnas (claves del json)
            self.columnas = [f"{self.titulo_plot[0]}",f"{self.titulo_plot[1]}"]

            # configuramos la tabla a partir de la información recuperada
            self.tableWidget.setRowCount(len(self.datos))
            # establecemos la longitud de las columnas
            self.tableWidget.setColumnCount(len(self.columnas))
            # establecemos las cabeceras de las columnas
            self.tableWidget.setHorizontalHeaderLabels(self.columnas)

            # dibujamos la tabla y los botones
            for i, fila in enumerate(self.datos):
                for j, columna in enumerate(self.columnas):
                    item = QTableWidgetItem()
                    # Con Qt.EditRole se establece el tipo de campo automáticamente
                    item.setData(Qt.EditRole, fila[columna])
                    self.tableWidget.setItem(i, j, item)

            # personalizamos y redimensionamos la tabla
            #self.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem(f"Signal {self.titulo_plot[0]}"))
            #self.tableWidget.setHorizontalHeaderItem(1, QTableWidgetItem(f"Signal {self.titulo_plot[1]}"))
            self.tableWidget.setVerticalHeaderItem(0, QTableWidgetItem(self.nombres_filas[0]))
            self.tableWidget.setVerticalHeaderItem(1, QTableWidgetItem(self.nombres_filas[1]))
            self.tableWidget.setVerticalHeaderItem(2, QTableWidgetItem(self.nombres_filas[2]))
            self.tableWidget.setVerticalHeaderItem(3, QTableWidgetItem(self.nombres_filas[3]))
            self.tableWidget.setVerticalHeaderItem(4, QTableWidgetItem(self.nombres_filas[4]))
            
             # PARA CREAR GRAFICO VISUAL
            self.grafica_powers.setBackground("w")
            powers = [self.r_seg_delta, self.r_seg_theta, self.r_seg_alpha, self.r_seg_beta, self.r_seg_gamma]
            middle = [0,0,0,0,0,0,0]
            powers_2 = [0,0,0,0,0,0,0,self.r_seg_delta_2, self.r_seg_theta_2, self.r_seg_alpha_2, self.r_seg_beta_2, self.r_seg_gamma_2]
            #names_powers = [f"Delta {self.titulo_plot[0]}",f"Theta {self.titulo_plot[0]}", f"Alpha {self.titulo_plot[0]}", 
            #f"Beta {self.titulo_plot[0]}", f"Gamma {self.titulo_plot[0]}"]

            bg1 = pg.BarGraphItem(x=range(5), height=powers, width=0.3, brush='b')
            bg_none = pg.BarGraphItem(x=range(7), height=middle, width=0.3, brush='b')
            bg2 = pg.BarGraphItem(x=range(12), height=powers_2, width=0.3, brush='orange')
            self.grafica_powers.addItem(bg1)
            self.grafica_powers.addItem(bg_none)
            self.grafica_powers.addItem(bg2)

            self.grafica_powers.setXRange(0,11)
            self.grafica_powers.setYRange(0,max(max(powers_2),max(powers)))

            #self.grafica_powers.setLabel("bottom",f'{self.titulo_plot[0]}  |  {self.titulo_plot[1]}',**styles)
           
            


        else:
            self.grafica_powers.clear()
            self.grafica_eeg.setTitle(f"{self.titulo_plot[0]}", size = "15px")
            #self.grafica_powers.setLabel("bottom",'Delta  -  Theta  -  Alpha  -  Beta  -  Gamma',**styles)  #f'{self.titulo_plot[0]}'

            self.nombres_filas = ["Ratio/Power delta", "Ratio/Power theta",
            "Ratio/Power alpha","Ratio/Power beta","Ratio/Power gamma"]
            
            # Creamos la estructura de datosX
            datos = []

            # Añadimos un registro manualmente
            datos.append({
                f"{self.s}": round(self.r_seg_delta,3),
            })

            # Definimos una lista de contactos
            contactos = [
                (round(self.r_seg_theta,3)),
                (round(self.r_seg_alpha,3)),
                (round(self.r_seg_beta,3)),
                (round(self.r_seg_gamma,3))
            ]

            # Añadimos un contactos dinámicamente
            for valor in contactos:
                datos.append({
                    f"{self.s}": valor,
                })

            # Guardamos los registros del fichero
            with open(absPath("contactos.json"), "w") as fichero:
                json.dump(datos, fichero)

            # Leemos los registros del fichero
            with open(absPath("contactos.json")) as fichero:
                datos = json.load(fichero)
                

            # cargamos el contenido del fichero
            with open(absPath("contactos.json")) as fichero:
                self.datos = json.load(fichero)

            # definimos la configuración de las columnas (claves del json)
            self.columnas = [f"{self.s}"]

            # configuramos la tabla a partir de la información recuperada
            self.tableWidget.setRowCount(len(self.datos))
            # establecemos la longitud de las columnas
            self.tableWidget.setColumnCount(len(self.columnas))
            # establecemos las cabeceras de las columnas
            self.tableWidget.setHorizontalHeaderLabels(self.columnas)

            # dibujamos la tabla y los botones
            for i, fila in enumerate(self.datos):
                for j, columna in enumerate(self.columnas):
                    item = QTableWidgetItem()
                    # Con Qt.EditRole se establece el tipo de campo automáticamente
                    item.setData(Qt.EditRole, fila[columna])
                    self.tableWidget.setItem(i, j, item)

            # personalizamos y redimensionamos la tabla
            self.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem(f"Signal {self.s}"))
            self.tableWidget.setVerticalHeaderItem(0, QTableWidgetItem(self.nombres_filas[0]))
            self.tableWidget.setVerticalHeaderItem(1, QTableWidgetItem(self.nombres_filas[1]))
            self.tableWidget.setVerticalHeaderItem(2, QTableWidgetItem(self.nombres_filas[2]))
            self.tableWidget.setVerticalHeaderItem(3, QTableWidgetItem(self.nombres_filas[3]))
            self.tableWidget.setVerticalHeaderItem(4, QTableWidgetItem(self.nombres_filas[4]))

            # PARA CREAR GRAFICO VISUAL
            self.grafica_powers.setBackground("w")
            powers = [self.r_seg_delta, self.r_seg_theta, self.r_seg_alpha, self.r_seg_beta, self.r_seg_gamma]
            names_powers = [f"Delta {self.titulo_plot[0]}",f"Theta {self.titulo_plot[0]}", f"Alpha {self.titulo_plot[0]}", 
            f"Beta {self.titulo_plot[0]}", f"Gamma {self.titulo_plot[0]}"]

            bg1 = pg.BarGraphItem(x=range(5), height=powers, width=0.3, brush='b')
            self.grafica_powers.addItem(bg1)
            self.grafica_powers.setXRange(0,4)
            self.grafica_powers.setYRange(0,max(powers))
            

    
            
    
    def show_tf(self):

        t = self.t
        signal = self.eeg[self.x,self.low:self.high]
        npts=len(signal)

        FFT = abs(fft(signal))

        freqs = fftfreq(npts, 1/self.fs)

        #calculamos los máximos de cada banda
        max_delta = np.max(FFT[np.abs(freqs-1.5).argmin():np.abs(freqs-3.5).argmin()+1])
        ind_delta = np.where(FFT == max_delta )[0][0]
        #ind_delta[0][0]

        max_theta = np.max(FFT[np.abs(freqs-3.5).argmin():np.abs(freqs-7.5).argmin()+1])
        ind_theta = np.where(FFT == max_theta )[0][0]
        #ind_delta[0][0]

        max_alpha = np.max(FFT[np.abs(freqs-7.5).argmin():np.abs(freqs-13.5).argmin()+1])
        ind_alpha = np.where(FFT == max_alpha)[0][0]
        #ind_delta[0][0]

        max_beta = np.max(FFT[np.abs(freqs-13.5).argmin():np.abs(freqs-30).argmin()+1])
        ind_beta = np.where(FFT == max_beta )[0][0]
        #ind_delta[0][0]

        frec_fund = [max_delta, max_theta, max_alpha, max_beta]
        ind_frec_fund = [ind_delta, ind_theta, ind_alpha, ind_beta]

        #plt.plot(self.tf_trozo[0], self.tf_trozo[1],'-', self.tf_trozo[0][ind_frec_fund],frec_fund,'x')
        
        frequencys = [round(freqs[ind_delta],2), round(freqs[ind_theta],2), round(freqs[ind_alpha],2), round(freqs[ind_beta],2)]
       # print("Las frec fundamentales son:",frequencys)


        self.grafica_tf.addLegend()
        self.grafica_tf.setBackground("w")

        if len(self.titulo_plot) == 2:
            self.tableWidget.clear()
            self.grafica_tf.clear()

            self.pos1 = self.signal_labels.index(str(self.titulo_plot[0]))
            self.pos2 = self.signal_labels.index(str(self.titulo_plot[1]))

            signal = self.eeg[self.pos1,self.low:self.high]
            signal2 = self.eeg[self.pos2,self.low:self.high]


            #calculamos su tf
            self.tf_trozo = abs(fft(signal))
            #self.tf_trozo = tf(self.eeg[self.pos1,self.low:self.high],1024,self.fs)

            max_delta = np.max(self.tf_trozo[np.abs(freqs-1.5).argmin():np.abs(freqs-3.5).argmin()+1])
            ind_delta = np.where(self.tf_trozo == max_delta )[0][0]
            #ind_delta[0][0]

            max_theta = np.max(self.tf_trozo[np.abs(freqs-3.5).argmin():np.abs(freqs-7.5).argmin()+1])
            ind_theta = np.where(self.tf_trozo == max_theta )[0][0]
            #ind_delta[0][0]

            max_alpha = np.max(self.tf_trozo[np.abs(freqs-7.5).argmin():np.abs(freqs-13.5).argmin()+1])
            ind_alpha = np.where(self.tf_trozo == max_alpha)[0][0]
            #ind_delta[0][0]

            max_beta = np.max(self.tf_trozo[np.abs(freqs-13.5).argmin():np.abs(freqs-30).argmin()+1])
            ind_beta = np.where(self.tf_trozo == max_beta )[0][0]
            #ind_delta[0][0]

    

            #calculamos su tf
            self.tf_trozo2 = abs(fft(signal2))
            #self.tf_trozo2 = tf(self.eeg[self.pos2,self.low:self.high],1024,self.fs)

            max_delta2 = np.max(self.tf_trozo2[np.abs(freqs-1.5).argmin():np.abs(freqs-3.5).argmin()+1])
            ind_delta2 = np.where(self.tf_trozo2 == max_delta2 )[0][0]
            #ind_delta[0][0]

            max_theta2 = np.max(self.tf_trozo2[np.abs(freqs-3.5).argmin():np.abs(freqs-7.5).argmin()+1])
            ind_theta2 = np.where(self.tf_trozo2 == max_theta2 )[0][0]
            #ind_delta[0][0]

            max_alpha2 = np.max(self.tf_trozo2[np.abs(freqs-7.5).argmin():np.abs(freqs-13.5).argmin()+1])
            ind_alpha2 = np.where(self.tf_trozo2 == max_alpha2)[0][0]
            #ind_delta[0][0]

            max_beta2 = np.max(self.tf_trozo2[np.abs(freqs-13.5).argmin():np.abs(freqs-30).argmin()+1])
            ind_beta2 = np.where(self.tf_trozo2 == max_beta2 )[0][0]
            #ind_delta[0][0]

            frec_fund2 = [max_delta2, max_theta2, max_alpha2, max_beta2]
            ind_frec_fund2 = [ind_delta2, ind_theta2, ind_alpha2, ind_beta2]

            frequencys1 = [round(freqs[ind_delta],2), round(freqs[ind_theta],2), round(freqs[ind_alpha],2), round(freqs[ind_beta],2)]
            frequencys2 = [round(freqs[ind_delta2],2), round(freqs[ind_theta2],2), round(freqs[ind_alpha2],2), round(freqs[ind_beta2],2)]
     

            #PARA REPRESENTAR TF
            self.grafica_tf.setTitle(f"FT - Dom. frec {self.titulo_plot[1]}: {frequencys1} - {self.titulo_plot[0]}: {frequencys2}", size = "12px")

            
            self.grafica_tf.plot(freqs[0:np.abs(freqs-30).argmin()+1], self.tf_trozo[0:np.abs(freqs-30).argmin()+1],pen= pg.mkPen("orange",width = 1), name= f"{self.titulo_plot[1]}")
            self.grafica_tf.plot(freqs[0:np.abs(freqs-30).argmin()+1], self.tf_trozo2[0:np.abs(freqs-30).argmin()+1],pen= pg.mkPen("blue",width = 1), name= f"{self.titulo_plot[0]}")

            self.grafica_tf.showGrid(x=True,y=True)
            #self.grafica_tf.addLegend()
        
        else:
            #PARA REPRESENTAR TF
            self.grafica_tf.clear()
            self.grafica_tf.setTitle(f"Dom. freq {self.s}: {frequencys}", size = "12px")
            self.grafica_tf.plot(freqs[0:np.abs(freqs-30).argmin()+1], FFT[0:np.abs(freqs-30).argmin()+1],pen= pg.mkPen("blue",width = 1), name= f"{self.s}")
            self.grafica_tf.showGrid(x=True,y=True)
            #self.grafica_tf.addLegend()
    

        styles = {"color":"#000", "font-size":"13px"}
        self.grafica_tf.setLabel("left",'Amplitude [dB]',**styles)
        self.grafica_tf.setLabel("bottom",'Frequency [Hz]',**styles)
  

        

    


    def show_psd(self):
        self.grafica_psd.addLegend()
        styles = {"color":"#000", "font-size":"13px"}
        self.grafica_psd.setLabel("left",'Amplitude',**styles)
        self.grafica_psd.setLabel("bottom",'Frequency [s]',**styles)
        self.grafica_psd.setBackground("w")

        if len(self.titulo_plot) == 2:
            self.grafica_psd.clear()
            
            #PARA REPRESENTAR PSD
            self.grafica_psd.setTitle(f"PSD {self.titulo_plot[0]} - {self.titulo_plot[1]}", size = "15px")
            self.x1 = self.signal_labels.index(self.titulo_plot[1])
            self.f1,self.p1 = ss.welch(self.eeg[self.x1, self.low:self.high], fs=self.fs )  #, nperseg=521, nfft=1024
            self.grafica_psd.plot(self.f1[0:31], self.p1[0:31], pen= pg.mkPen("6E6DE3",width = 1),  name= f"Segment {self.titulo_plot[0]}")
            #represento la psd de la señal entera para comparar
            self.f1_total,self.p1_total = ss.welch(self.eeg[self.x1, :], fs=self.fs)
            self.grafica_psd.plot(self.f1_total[0:31], self.p1_total[0:31], pen= pg.mkPen("0100FF",width = 1),  name= f"{self.titulo_plot[0]}")

            # SEÑAL 1 vs SEÑAL 2

            self.x2 = self.signal_labels.index(self.titulo_plot[0])
            self.f2,self.p2 = ss.welch(self.eeg[self.x2, self.low:self.high], fs=self.fs)  #, nperseg=521, nfft=1024
            self.grafica_psd.plot(self.f2[0:31],self.p2[0:31], pen= pg.mkPen("orange",width = 1), name= f"Segment {self.titulo_plot[1]}")

            #represento la psd de la señal entera para comparar
            self.f2_total,self.p2_total = ss.welch(self.eeg[self.x2, :], fs=self.fs)
            self.grafica_psd.plot(self.f2_total[0:31], self.p2_total[0:31], pen= pg.mkPen("orange",width = 1), name= f"{self.titulo_plot[1]}")

            self.grafica_psd.showGrid(x=True,y=True)
            #self.grafica_psd.addLegend()
    
        
        else:
            #PARA REPRESENTAR PSD
            self.grafica_psd.clear()

            #represento la psd de la señal entera para comparar
            self.grafica_psd.setTitle(f"PSD {self.s}", size = "15px")
            self.f,self.p = ss.welch(self.eeg[self.x, :], fs=self.fs) # , nperseg=521, nfft=1024)
            self.grafica_psd.plot(self.f[0:31],self.p[0:31], pen= pg.mkPen("#0100FF",width = 1), name= f"{self.s}")

            # represento el trozo elegido
            self.f_total,self.p_total = ss.welch(self.eeg[self.x, self.low:self.high], fs=self.fs)
            self.grafica_psd.plot(self.f_total[0:31], self.p_total[0:31], pen= pg.mkPen("#6E6DE3",width = 1), name= f"Segment {self.titulo_plot[0]}")

            self.grafica_psd.showGrid(x=True,y=True)
            #self.grafica_psd.addLegend()

    


    def show_PSDWav(self):
        self.grafica_spectgram.clear()
        x1 = self.signal_labels.index(self.titulo_plot[0])

        #wavelet = "db2"
        #level = 4
        #order = "freq"  # other option is "normal"
        #wp = pywt.WaveletPacket(self.eeg[x1,self.low:self.high], wavelet, 'symmetric', maxlevel=level)
        #nodes = wp.get_level(level, order=order)
        #labels = [n.path for n in nodes]
        #values = np.array([n.data for n in nodes], 'd')
        #wPSD = abs(values)

        coef, freqs= pywt.cwt(self.eeg[x1,self.low:self.high],np.arange(0.1,self.fs/2  + 1), self.wav_select)
        
        #f0 = np.linspace(0.1, 25, 100)
        #wPSD = wavelet_PSD(self.t[self.low:self.high], self.eeg[x1,self.low:self.high], f0)

        #self.grafica_spectgram.setTitle(f"PSD-Wav {self.s}", size = "20px")
        view = self.grafica_spectgram.addViewBox()
      
        #pcmi = pg.PColorMeshItem(wPSD[0:30,].T)  # de 0 a 30 las frecuencias
        pcmi = pg.PColorMeshItem(coef.T)  # de 0 a 60 las frecuencias
        view.addItem(pcmi)

        


    def show_spectgram(self):
        self.grafica_spectgram.clear()
        x1 = self.signal_labels.index(self.titulo_plot[0])
        #f, t_spect, Sxx = signal.spectrogram(self.eeg[x1,self.low:self.high], fs=self.fs, nfft=512)

        nperseg = self.fs
        ftf, ftt, ftZ = signal.stft(self.eeg[x1,self.low:self.high] , nperseg=nperseg, fs=self.fs)
       

        #self.grafica_spectgram.addLabel("top","Spectrogram")
        view = self.grafica_spectgram.addViewBox()

        #pcmi = pg.PColorMeshItem(Sxx.T)
        pcmi = pg.PColorMeshItem(np.abs(ftZ[0:30].T))    # de 0 a 30 las frecuencias
        
        view.addItem(pcmi)



    def generar_pdf(self):
       
        try:
            # Dataframe de la tabla

            self.coments = self.textEdit.toPlainText()  #para quedarnos con el contenido del recuadro de comentarios
            
            # PRIMERO GENERAR HTML Y LUEGO GENERAR EL PDF A PARTIR DE DICHO HTML
            
            
            df = pd.DataFrame(self.datos ,  index = self.nombres_filas)
            #print(df)

            env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=absPath("plantillas")))
            template = env.get_template("template.html")

            styler = df.style.applymap(lambda valor: "color:black" if valor > 0.5 else "color:black")

            html = template.render(tabla = styler.render())
            
            with open(absPath("reporte.html"),"w") as f:
                f.write(html)
                f.write(self.coments)
                f.write("                                        ")
                r1 = str(self.ratio_1.currentText())
                r2 = str(self.ratio_2.currentText())
                r_total = self.ratio_selected
                f.write(f" - Ratio {r1}/{r2}: {r_total} ")
                

            # exportar los graficos a imagenes
            exporter = pg.exporters.ImageExporter(self.grafica_eeg.plotItem)
            exporter.export(absPath("eeg_plot.png"))

            exporter = pg.exporters.ImageExporter(self.grafica_psd.plotItem)
            exporter.export(absPath("psd_plot.png"))

            exporter = pg.exporters.ImageExporter(self.grafica_spectgram.sceneObj)  # psdWav o espectrobgrama
            exporter.export(absPath("spectgram_plot.png"))

            exporter = pg.exporters.ImageExporter(self.grafica_tf.plotItem)
            exporter.export(absPath("tf_plot.png"))

            exporter = pg.exporters.ImageExporter(self.grafica_pendiente.plotItem)
            exporter.export(absPath("slope_plot.png"))

            exporter = pg.exporters.ImageExporter(self.grafica_powers.plotItem)  # la grafica de barras visual de las potencias
            exporter.export(absPath("powers_bars_plot.png"))

            #exporter = pg.exporters.ImageExporter(self.textEdit.insertHtml(html))  # lo apuntado
            #exporter.export(absPath("resumen.png"))

            # origen - destino
            options = {"enable-local-file-access": None}
            pdfkit.from_file(absPath("reporte.html"), absPath("reporte.pdf"), options = options)

            os.startfile(absPath("reporte.pdf"),"open")
        
        # por si hubiese un error generando el pdf
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ups", f"Error generando el reporte \n\n{e}")

        #else:
            #self.statusBar.showMessage("Reporte generado correctamente.")
    
    
    def show_ica(self):

        if self.eeg_object.info["dig"] == None:    # si no hay posiciones de electrodos disponibles

            biosemi_montage = mne.channels.make_standard_montage('standard_1020')

            new_positions = []
            electrodes = [1,2,3  ,15,17,19,21,23,  86,39,   92,93,41,43,80,    82,61,63,65,89,   88,87]

            # primera iteracion
            for i in biosemi_montage.dig[6:-1]:
            #print(i["ident"])
                if i["ident"] not in electrodes:
                    pass
                else:
                    new_positions.append(biosemi_montage.dig[3 + i["ident"]])
                    #eeg.info["dig"] = raw.info["dig"][85 + i["ident"]]

            # segunda iteracion debido al error que hay en los nombres de las posiciones
            new_positions_2 = []
            n = 0
            for i in range(3):
                new_positions_2.append(biosemi_montage.dig[i+3])
                n+=1

            # 3º iteracion
            for i in new_positions[0:-1]:
                new_positions_2.append(i)

            new_positions_2.append(new_positions[-1])

        

            # para quedarnos con los electrodos de tipo eeg que usa el neurologo en caso de que no hayan puntos de los electr especificados
            signals = self.eeg

            ch_types = ['eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg','eeg', 'eeg']
            ch_names = self.eeg_object.ch_names[0:22]
            data = np.array([signals[0,:], signals[1,:], signals[2,:], signals[3,:], signals[4,:], signals[5,:], signals[6,:], signals[7,:], signals[8,:], signals[9,:], signals[10,:], signals[11,:], 
                            signals[12,:], signals[13,:], signals[14,:], signals[15,:],signals[16,:], signals[17,:], signals[18,:], signals[19,:],signals[20,:], signals[21,:]])

            info = mne.create_info(ch_names=ch_names, sfreq=self.fs, ch_types=ch_types)   # info de eeg con los 22 canales

            self.eeg_ica = mne.io.RawArray(data, info)

            names_electrodes = ['Fp1' , 'Fpz' , 'Fp2', "F7" ,'F3', 'Fz', 'F4', 'F8', 'A1',"A2", 'T3', 'C3', 'Cz', 'C4',"T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
            ELECTRODES = mne.channels.DigMontage(dig=new_positions_2, ch_names= names_electrodes)

            # le pasamos el montage al archivo
            self.eeg_ica.set_montage(ELECTRODES)
        

        else:         # si si que hay posiciones de electrodos incluidas en el archivo
            pass

        
        # CALCULAMOS ICA
        n_comps = 20
        ica = ICA(n_components=n_comps, random_state=97)
        self.comps = ica.fit(self.eeg_ica)   

        icas_comp_raw = self.comps.get_sources(self.eeg_ica)  # para coger las ICA proporcionadas 1 a 1
        icas_comp_raw = icas_comp_raw.get_data()     # para convertirlas a data para poder leerlas

        #creamos la subventana y representamos las señales ICA
        plt_ica = pg.plot()
        plt_ica.setWindowTitle('ICA components')
        #plt.setLabel('bottom', 'Index', units='B')

        legend = pg.LegendItem( offset=(-600,-20), verSpacing= 1)
        legend.setParentItem(plt_ica.graphicsItem())

        nPlots = n_comps
        curves = []
        for idx in range(nPlots):
            curve = pg.PlotCurveItem(pen=({'color': "black", 'width': 1}), skipFiniteCheck=True, name = f"Signal {idx} ")
            plt_ica.addItem(curve)
            curve.setPos(0,-idx*6)
            curves.append(curve)
            legend.addItem(curve, f"ICA-{idx} ")
            

        n = 0
        for i in range(nPlots):
            curves[i].setData(self.t, icas_comp_raw[i,:] - n)
            n += 0.005
        
        
        plt_ica.setBackground("w")
        plt_ica.setXRange(0,len(self.t)/self.fs + 3)

        plt_ica.resize(1500,600)


        #  CREAMOS LA IMAGEN DE LOS CEREBROS Y LO REPRESENTAMOS
        brains = self.comps.plot_components()   



    def filter_ica(self):
        try:
            to_exclude = []
            to_exclude_ICA = self.to_exclude_ICA.text().split(",")
            for i in to_exclude_ICA:
                    to_exclude.append(int(i))
                
            print(to_exclude)
            self.comps.exclude = to_exclude
            self.comps.apply(self.eeg_ica)

            self.eeg_ica_final = self.eeg_ica.get_data()

            self.number = 0

            QtWidgets.QMessageBox.information(self, "Success", f"Signals have been filtered successfully.")
        
        except ValueError: 
            QtWidgets.QMessageBox.critical(self, "Ups.", f"Error filtrando las señales, revise que ha indicado las ICA correctamente.")

        






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
