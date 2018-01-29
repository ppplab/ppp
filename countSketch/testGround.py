# Test hash functions
import hashlib
from bitarray import bitarray
from bitstring import BitArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import config

def testHash():
    # Conver to SHA-256 hash
    m = hashlib.sha256()
    m.update('Abhradee Guha Thakurta')
    mInBytes=m.digest()

    # Retreive an integer for a sequence of hash bits
    a = bitarray(endian = 'little')
    a.frombytes(mInBytes)
    print BitArray(a[1:10]).uint

def drawGroupedBarGraph(dataFrame, xGroups, alpha = 0.5, label = '', yLabel = '', title =''):
    pos = list(range(len(dataFrame[[0]])))
    fig, ax = plt.subplots(figsize=(10, 5))
    numGroups = len(xGroups)
    width = 1.0 / (numGroups + 1)
    maxColor = 250
    for i in range(numGroups):
        print dataFrame.columns[1+2*i]
        plt.bar([p + i*width for p in pos],dataFrame[dataFrame.columns[1+2*i]],width, alpha=0.5, color = cm.Greys(maxColor - i* 40),yerr = dataFrame[dataFrame.columns[2+2*i]],error_kw=dict(capsize=5))

    ax.set_ylabel(yLabel)
    ax.set_title(title)

    ax.set_xticks([p + (len(xGroups)/2.0) * width for p in pos])
    ax.set_xticklabels(dataFrame[dataFrame.columns[0]])

    plt.xlim(min(pos) - width, max(pos) + width * (numGroups +1))
    maxMean = 0
    for i in range(0,(len(dataFrame.columns)-1)/2):
        if maxMean < max(dataFrame[dataFrame.columns[1+2*i]]): maxMean = max(dataFrame[dataFrame.columns[1+2*i]])

    print maxMean
    plt.ylim([0, 1.8 * maxMean])
    plt.legend(xGroups, loc='upper left')
    plt.grid()
    plt.show()

def drawGraph():
    file = pd.read_csv(config.config.dataPath+'BrownDataSet.csv')
    counts = np.array(file['trueFrequency'].tolist()[0:100]).astype('float32')
    counts = counts / np.sum(counts)
    print counts
    y_pos = np.arange(len(counts))

    plt.grid()
    plt.bar(y_pos, counts, align='center',color='b')
    #plt.xticks(y_pos, objects)
    plt.ylabel('Normalized Frequency')
    plt.xlabel('Words')

    plt.show()

def testMatPlotLib():
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                'pre_score': [4, 24, 31, 2, 3],
                'pre_score_std': [4, 2, 3, 2, 3],
                'mid_score': [25, 94, 57, 62, 70],
                'mid_score_std': [2, 9, 5, 6, 7]
                }

    df = pd.DataFrame(raw_data, columns=['first_name','pre_score','pre_score_std','mid_score','mid_score_std'])
    print df
    drawGroupedBarGraph(df, ['Pre Score', 'Mid Score'], yLabel = 'Score', title = 'Blah')

drawGraph()
