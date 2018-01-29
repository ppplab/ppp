# The core logic for private count sketch [Charikar-Chen-Farach-Colton 2004]
import config
import numpy as np
import hashlib
from bitarray import bitarray
from bitstring import BitArray
import DJWRandomizer
import math

class countSketch:
    sketchMatrix = np.zeros((config.config.l, config.config.w))

    @staticmethod
    def getSHA256HashArray(hashId, dataString):
        message = hashlib.sha256()

        message.update(str(hashId) + dataString)
        messageInBytes = message.digest()

        messageInBitArray = bitarray(endian='little')
        messageInBitArray.frombytes(messageInBytes)

        return messageInBitArray

    @staticmethod
    def setSketchElement(dataString):
        assert (isinstance(dataString, str) == True), 'Data should be a string'

        hashId = np.random.randint(0,config.config.l)
        messageInBitArray = countSketch.getSHA256HashArray(hashId, dataString)

        hLoc = BitArray(messageInBitArray[0: int(math.log(config.config.w, 2))]).uint
        gVal = 2 * messageInBitArray[int(math.log(config.config.w, 2))] - 1

        dataVec = np.zeros(config.config.w)
        dataVec[hLoc] = gVal

        privatizedVec = DJWRandomizer.randomize(dataVec)

        countSketch.sketchMatrix[hashId]+= (privatizedVec * config.config.cEpsilon * config.config.l)


    @staticmethod
    def writeSketchToFile(sketchLocation):
        np.save(config.config.dataPath + sketchLocation, countSketch.sketchMatrix)

    @staticmethod
    def readSketch(sketchLocation):
        countSketch.sketchMatrix = np.load(config.config.dataPath + sketchLocation)

    @staticmethod
    def getFreqEstimate(dataString):
        assert (isinstance(dataString, str) == True), 'Data should be a string'

        weakFreqEstimates = np.zeros(config.config.l)
        for hashId in range(0, config.config.l):
            messageInBitArray = countSketch.getSHA256HashArray(hashId, dataString)

            hLoc = BitArray(messageInBitArray[0: int(math.log(config.config.w, 2))]).uint
            gVal = 2 * messageInBitArray[int(math.log(config.config.w, 2))] - 1
            weakFreqEstimates[hashId] = gVal * countSketch.sketchMatrix[hashId, hLoc]
        estimate = np.median(weakFreqEstimates)
        return estimate if estimate >0 else 0













