# This Python file uses the following encoding: utf-8
import sys
import os
import tempfile
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsProxyWidget
from PySide6.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from processData import DataProcessor


input_folder = r"D:\PPGFilteringProject\acquisition_test\acquisition_test"
sample_name = "PPG_IR_S016_finger.npy"
default_fs = 2000
fitting_type = "Gauss"
use_filter = True
cutoff_low = 0.05
cutoff_high = 10
filter_order = 2

# create istance of class processData
processor = DataProcessor(
    filepath=os.path.join(input_folder,sample_name),
    fs=default_fs,
    fitting_type=fitting_type,
    use_filter=use_filter,
    cutoff_low=cutoff_low,
    cutoff_high=cutoff_high,
    filter_order=filter_order
)

parameters, R2, NRMSE, MSE, figure = processor.process_signal()
        


