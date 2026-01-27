import sys
sys.path.append('emulator/')

### GENERAL TOOLS ###
import os
import json
import pickle
import copy
import glob
import re
import bisect
import trimesh
import colorsys
from tqdm import tqdm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from typing import Callable, cast
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from datetime import datetime, timezone
from collections import deque

### PYQT ###
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QToolBar, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QScrollArea, QListWidget, QListWidgetItem,
    QSizePolicy, QComboBox, QLabel, QFrame, QMessageBox, QSlider,
    QRadioButton, QCheckBox, QButtonGroup, QProgressBar
)
from QMarkdownWidget import QMView

### PLOTTING ###
import plotly.graph_objects as go
import plotly.io as pio
import plotly
import matplotlib
matplotlib.use( 'QtAgg' )
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure