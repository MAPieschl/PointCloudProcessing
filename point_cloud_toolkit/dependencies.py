import sys
sys.path.append('emulator/')

### GENERAL TOOLS ###
import os
import json
import pickle
import copy
import glob
import numpy as np
from typing import Callable
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

### PYQT ###
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QToolBar, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QScrollArea, QListWidget, QListWidgetItem,
    QSizePolicy, QComboBox, QLabel, QFrame, QMessageBox, QSlider,
    QRadioButton, QCheckBox
)
from QMarkdownWidget import QMView

### PLOTTING ###
import plotly.graph_objects as go
import plotly.io as pio
import plotly

### GLOBALS ###
import utils.globals as globals
import utils.mat_ops as mat_ops
import utils.corner_reflector as corner_reflector