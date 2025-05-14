# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QFrame, QGraphicsView, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QListView, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QSizePolicy,
    QSplitter, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1015, 683)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Widget.sizePolicy().hasHeightForWidth())
        Widget.setSizePolicy(sizePolicy)
        self.formLayout = QFormLayout(Widget)
        self.formLayout.setObjectName(u"formLayout")
        self.splitter = QSplitter(Widget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.groupBox_6 = QGroupBox(self.splitter)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setAlignment(Qt.AlignCenter)
        self.gridLayout_6 = QGridLayout(self.groupBox_6)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.open_input_folder = QPushButton(self.groupBox_6)
        self.open_input_folder.setObjectName(u"open_input_folder")

        self.gridLayout_6.addWidget(self.open_input_folder, 0, 0, 1, 1)

        self.list_valid_files = QListWidget(self.groupBox_6)
        self.list_valid_files.setObjectName(u"list_valid_files")
        self.list_valid_files.setStyleSheet(u"background-color:white")
        self.list_valid_files.setFrameShape(QFrame.Box)
        self.list_valid_files.setFrameShadow(QFrame.Plain)
        self.list_valid_files.setLayoutMode(QListView.Batched)
        self.list_valid_files.setBatchSize(20)

        self.gridLayout_6.addWidget(self.list_valid_files, 1, 0, 1, 1)

        self.splitter.addWidget(self.groupBox_6)
        self.groupBox_7 = QGroupBox(self.splitter)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setAlignment(Qt.AlignCenter)
        self.gridLayout_7 = QGridLayout(self.groupBox_7)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.fs_line_edit = QLineEdit(self.groupBox_7)
        self.fs_line_edit.setObjectName(u"fs_line_edit")

        self.gridLayout_7.addWidget(self.fs_line_edit, 0, 0, 1, 1)

        self.splitter.addWidget(self.groupBox_7)
        self.groupBox_8 = QGroupBox(self.splitter)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setAlignment(Qt.AlignCenter)
        self.gridLayout_8 = QGridLayout(self.groupBox_8)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gauss_fit_checked = QRadioButton(self.groupBox_8)
        self.gauss_fit_checked.setObjectName(u"gauss_fit_checked")

        self.gridLayout_8.addWidget(self.gauss_fit_checked, 0, 0, 1, 1)

        self.exp_fit_checked = QRadioButton(self.groupBox_8)
        self.exp_fit_checked.setObjectName(u"exp_fit_checked")
        self.exp_fit_checked.setChecked(True)

        self.gridLayout_8.addWidget(self.exp_fit_checked, 1, 0, 1, 1)

        self.splitter.addWidget(self.groupBox_8)
        self.groupBox_9 = QGroupBox(self.splitter)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.groupBox_9.setEnabled(True)
        self.groupBox_9.setAlignment(Qt.AlignCenter)
        self.groupBox_9.setCheckable(False)
        self.gridLayout_9 = QGridLayout(self.groupBox_9)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.low_cutoff = QLineEdit(self.groupBox_9)
        self.low_cutoff.setObjectName(u"low_cutoff")

        self.gridLayout_9.addWidget(self.low_cutoff, 4, 1, 1, 1)

        self.high_cutoff = QLineEdit(self.groupBox_9)
        self.high_cutoff.setObjectName(u"high_cutoff")

        self.gridLayout_9.addWidget(self.high_cutoff, 4, 2, 1, 1)

        self.use_filter_button = QRadioButton(self.groupBox_9)
        self.use_filter_button.setObjectName(u"use_filter_button")
        self.use_filter_button.setCheckable(True)
        self.use_filter_button.setChecked(False)

        self.gridLayout_9.addWidget(self.use_filter_button, 0, 0, 5, 1)

        self.label_6 = QLabel(self.groupBox_9)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.gridLayout_9.addWidget(self.label_6, 3, 1, 1, 1)

        self.label_7 = QLabel(self.groupBox_9)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignBottom|Qt.AlignHCenter)

        self.gridLayout_9.addWidget(self.label_7, 3, 2, 1, 1)

        self.order_box = QComboBox(self.groupBox_9)
        self.order_box.addItem("")
        self.order_box.addItem("")
        self.order_box.addItem("")
        self.order_box.addItem("")
        self.order_box.addItem("")
        self.order_box.setObjectName(u"order_box")
        self.order_box.setEditable(True)
        self.order_box.setMaxVisibleItems(4)

        self.gridLayout_9.addWidget(self.order_box, 5, 2, 1, 1)

        self.label_8 = QLabel(self.groupBox_9)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_9.addWidget(self.label_8, 5, 1, 1, 1)

        self.splitter.addWidget(self.groupBox_9)
        self.groupBox_10 = QGroupBox(self.splitter)
        self.groupBox_10.setObjectName(u"groupBox_10")
        self.groupBox_10.setAlignment(Qt.AlignCenter)
        self.gridLayout_10 = QGridLayout(self.groupBox_10)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.exportParameters_2 = QCheckBox(self.groupBox_10)
        self.exportParameters_2.setObjectName(u"exportParameters_2")

        self.gridLayout_10.addWidget(self.exportParameters_2, 0, 0, 1, 2)

        self.output_dir_button = QPushButton(self.groupBox_10)
        self.output_dir_button.setObjectName(u"output_dir_button")
        icon = QIcon()
        icon.addFile(u"../../../python_exp_processing/folder_into.png", QSize(), QIcon.Normal, QIcon.Off)
        self.output_dir_button.setIcon(icon)

        self.gridLayout_10.addWidget(self.output_dir_button, 2, 1, 1, 1)

        self.output_path = QLineEdit(self.groupBox_10)
        self.output_path.setObjectName(u"output_path")

        self.gridLayout_10.addWidget(self.output_path, 2, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox_10)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_10.addWidget(self.label_2, 1, 0, 1, 1)

        self.splitter.addWidget(self.groupBox_10)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.splitter)

        self.splitter_2 = QSplitter(Widget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Vertical)
        self.fit_all_button = QPushButton(self.splitter_2)
        self.fit_all_button.setObjectName(u"fit_all_button")
        self.splitter_2.addWidget(self.fit_all_button)
        self.display_plot = QGraphicsView(self.splitter_2)
        self.display_plot.setObjectName(u"display_plot")
        self.display_plot.setEnabled(True)
        sizePolicy.setHeightForWidth(self.display_plot.sizePolicy().hasHeightForWidth())
        self.display_plot.setSizePolicy(sizePolicy)
        self.splitter_2.addWidget(self.display_plot)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.splitter_2)

        self.label_9 = QLabel(Widget)
        self.label_9.setObjectName(u"label_9")
        font = QFont()
        font.setPointSize(12)
        self.label_9.setFont(font)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_9)

        self.console = QListWidget(Widget)
        self.console.setObjectName(u"console")
        sizePolicy.setHeightForWidth(self.console.sizePolicy().hasHeightForWidth())
        self.console.setSizePolicy(sizePolicy)
        self.console.setMaximumSize(QSize(16777215, 123))

        self.formLayout.setWidget(2, QFormLayout.SpanningRole, self.console)


        self.retranslateUi(Widget)
        self.open_input_folder.clicked.connect(Widget.select_input_folder)
        self.output_dir_button.clicked.connect(Widget.select_output_folder)
        self.list_valid_files.itemDoubleClicked.connect(Widget.process_file)

        self.order_box.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("Widget", u"Input Data", None))
        self.open_input_folder.setText(QCoreApplication.translate("Widget", u"Open Folder", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("Widget", u"Sampling Freq (Hz)", None))
        self.fs_line_edit.setText(QCoreApplication.translate("Widget", u"60", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("Widget", u"Fitting Function", None))
        self.gauss_fit_checked.setText(QCoreApplication.translate("Widget", u"Gaussian", None))
        self.exp_fit_checked.setText(QCoreApplication.translate("Widget", u"Exponential", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("Widget", u"Butterworth Bandpass Filter", None))
        self.low_cutoff.setText("")
        self.high_cutoff.setText("")
        self.use_filter_button.setText(QCoreApplication.translate("Widget", u"Use filter", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"Cutoff low", None))
        self.label_7.setText(QCoreApplication.translate("Widget", u"Cutoff high", None))
        self.order_box.setItemText(0, QCoreApplication.translate("Widget", u"None", None))
        self.order_box.setItemText(1, QCoreApplication.translate("Widget", u"1", None))
        self.order_box.setItemText(2, QCoreApplication.translate("Widget", u"2", None))
        self.order_box.setItemText(3, QCoreApplication.translate("Widget", u"3", None))
        self.order_box.setItemText(4, QCoreApplication.translate("Widget", u"4", None))

        self.label_8.setText(QCoreApplication.translate("Widget", u"Order", None))
        self.groupBox_10.setTitle(QCoreApplication.translate("Widget", u"Output Data", None))
        self.exportParameters_2.setText(QCoreApplication.translate("Widget", u"Export Parameters", None))
        self.output_dir_button.setText("")
        self.label_2.setText(QCoreApplication.translate("Widget", u"Output Folder", None))
        self.fit_all_button.setText(QCoreApplication.translate("Widget", u"Fit All Data", None))
        self.label_9.setText(QCoreApplication.translate("Widget", u"Console", None))
    # retranslateUi

