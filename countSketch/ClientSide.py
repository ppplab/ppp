# Privatize the data sample using the count sketch
import config
import PrivCountSketch
import numpy as np

def privatizeData(dataString):
    PrivCountSketch.countSketch.setSketchElement(dataString)

