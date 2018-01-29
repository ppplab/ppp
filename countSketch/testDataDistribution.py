# Play with word distributions from NTLK and simulation
from nltk.corpus import brown
from collections import Counter
from string import lower
from itertools import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config
import PrivCountSketch
import ServerSide
import math
import uuid

# Use the next three lines only on Mac
import resource
rsrc = resource.RLIMIT_DATA
resource.setrlimit(rsrc, (20971520, 41943040))

def genRandomWords(numberOfWords,numOfCharacters):
    assert numberOfWords <= math.pow(26,numOfCharacters), 'Too many words requested for'
    dict = {}
    while len(dict.keys()) < numberOfWords:
        randomNumbers = np.random.randint(97,123,numOfCharacters)
        candidateWord = ''.join(chr(i) for i in randomNumbers)

        if candidateWord in dict.keys():
            continue

        dict[candidateWord] = 0
    return dict.keys()
def truncateOrExtendWord(currentWord):
    wordLength = config.config.numNgrams * config.config.gramLength
    lastChar = currentWord[-1]
    if len(currentWord) <= wordLength:
        currentWord+= lastChar * (wordLength-len(currentWord))
    else:
        currentWord = currentWord[:wordLength]
    return currentWord

# Plotting tool
def drawGroupedBarGraph(dataFrame, xGroups, filePrefix, alpha = 0.5, label = '', yLabel = '', title =''):
    numGroups = len(dataFrame[[0]])
    print numGroups
    pos = list(range(numGroups))
    width = 1.0 / (2 * len(xGroups) + 1)
    fig, ax = plt.subplots(figsize=(10,5))
    my_colors = 'rgbkymc'
    for i in range(len(xGroups)):
        print dataFrame.columns[1+len(xGroups)*i]
        plt.bar([p+2*i*width for p in pos], dataFrame[dataFrame.columns[1+3*i]],width,alpha=0.5,
                color = my_colors[2*i])
        plt.bar([p + 2*i * width + width for p in pos], dataFrame[dataFrame.columns[2+3*i]], width, alpha=0.5,
                color = my_colors[2*i+1],yerr = dataFrame[dataFrame.columns[3+3*i]],error_kw=dict(capsize=5))
    ax.set_ylabel(yLabel)
    ax.set_title(title)

    ax.set_xticks([p + (len(xGroups)) * width for p in pos])
    ax.set_xticklabels(dataFrame[dataFrame.columns[0]])

    plt.xlim(min(pos) - width, max(pos) + width * (2*len(xGroups))+width)
    maxMean = 0
    for i in range(len(xGroups)):
        if maxMean < max(dataFrame[dataFrame.columns[1+3*i]]): maxMean = max(
            dataFrame[dataFrame.columns[1 + 3 * i]])
        if maxMean < max(dataFrame[dataFrame.columns[1 + 3 * i]]): maxMean = max(
            dataFrame[dataFrame.columns[2 + 3 * i]])

    print maxMean
    plt.ylim([0, 2.0 * maxMean])
    xLeg = []
    for s in xGroups:
        xLeg.append(s+'-True')
        xLeg.append(s + '-Priv')

    plt.legend(xLeg, loc='upper left')
    plt.grid()
    plt.savefig(config.config.dataPath + filePrefix +'_frequency.pdf',bbox_inches='tight')

# Create data set
def runWithBrownCorpursNLTK(filePrefix = '', n = config.config.n, epsilon = config.config.epsilon, w = config.config.w, threshold = config.config.threshold, wordDiscovery = False):
    config.config.n = n
    config.config.epsilon = epsilon
    config.config.w = w
    config.config.threshold = threshold
    config.config.cEpsilon = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)  # Scaling for de-biasing
    config.config.threshold = 15.0 * int(math.sqrt(n)) # Threshold for discoverability
    assert (int(math.log(w, 2)) <= 254), 'Sketch size (w) too large'

    wordsWithAlphabetsAndLower = [s.lower() for s in brown.words() if s.isalpha()]
    baseDistrLength = len(wordsWithAlphabetsAndLower)
    randNumbers = np.random.randint(0,baseDistrLength, config.config.n)
    datSet = [truncateOrExtendWord(str(wordsWithAlphabetsAndLower[x])) for x in randNumbers]
    print 'Data set creation completed'

    # The data: token counts from the Brown corpus
    tokens_with_count = sorted(Counter(imap(lower, datSet)).items(),  key=lambda i: i[1], reverse = True)
    wordFrequency = pd.DataFrame(tokens_with_count, columns=['word','trueFrequency'])

    #Abhradeep hack: Write the corpus to a file
    wordFrequency.to_csv(config.config.dataPath+'BrownDataSet.csv', sep = ',')
    configFileName = 'configBrown'+'_numWords='+str(wordFrequency.__len__())+'.txt'
    reportFileName = configFileName.replace('.txt','.csv').replace('config','corpus')

    if wordDiscovery:
        ServerSide.runServerSideWordDiscovery(wordFrequency, configFileName, reportFileName, filePrefix=filePrefix)
    else:
        ServerSide.runServerSide(wordFrequency, configFileName, reportFileName, filePrefix=filePrefix)
    return [configFileName, reportFileName]

def runWithPowerLawDist(filePrefix = '', n = config.config.n, epsilon = config.config.epsilon, w = config.config.w, threshold = config.config.threshold, wordDiscovery = False):
    config.config.n = n
    config.config.epsilon = epsilon
    config.config.w = w
    config.config.threshold = threshold
    config.config.cEpsilon = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)  # Scaling for de-biasing
    config.config.threshold = 15.0 * int(math.sqrt(n)) # Threshold for discoverability
    assert (int(math.log(w, 2)) <= 254), 'Sketch size (w) too large'

    a = 15.  # shape
    samples = config.config.n
    numWords = 100
    wordLength = config.config.gramLength *config.config.numNgrams
    s = np.random.power(a, samples)
    count, bins, ignored = plt.hist(s, bins=numWords)
    plt.close()

    wordList = genRandomWords(len(count), wordLength)
    tokens_with_count = sorted([(wordList[i], int(count[i])) for i in range(len(count))], key = lambda i:i[1], reverse=True)
    wordFrequency = pd.DataFrame(tokens_with_count, columns=['word', 'trueFrequency'])
    wordFrequency = wordFrequency[wordFrequency.trueFrequency != 0]
    #Abhradeep hack: Write the corpus to a file
    wordFrequency.to_csv(config.config.dataPath+'powerLawDataSet.csv', sep = ',')

    print 'Max frequency =', np.max(wordFrequency['trueFrequency'])
    configFileName = 'configPowerLaw_power='+str(a).replace('.','dec')+'_numWords='+str(wordFrequency.__len__())+'_wordLength='+str(wordLength)+'.txt'
    reportFileName = configFileName.replace('.txt','.csv').replace('config','corpus')

    if wordDiscovery:
        ServerSide.runServerSideWordDiscovery(wordFrequency, configFileName, reportFileName, filePrefix = filePrefix)
    else:
        ServerSide.runServerSide(wordFrequency, configFileName, reportFileName, filePrefix=filePrefix)

    return [configFileName, reportFileName]


# Experiment 1: Run with varying values of n
def runFrequencyEstimationWithVaryingN():
    config.config.reinitializeParameters()
    nChoices = [100000,1000000,5000000,10000000]
    displayLabels = ['50pctle','90pctle','95pctle']
    columns = ['n','50pctle-True','50pctle-Priv','50pctle-Priv-Std','90pctle-True','90pctle-Priv','90pctle-Priv-Std','95pctle-True','95pctle-Priv','95pctle-Priv-Std']
    resutDataFramne = pd.DataFrame(columns = columns)
    for y,i in enumerate(nChoices):
        print 'Running expt for n=', i
        filePrefix = str(uuid.uuid4()) + '_'
        w = config.config.nearestPowerOfTwoCeil(math.sqrt(i))

        [configFileName, reportFileName] = runWithPowerLawDist(filePrefix=filePrefix, n=i, w=w)
        dataFrame = pd.read_csv(config.config.dataPath + filePrefix + reportFileName)

        trueFreq = dataFrame['trueFrequency']
        privteFreq = dataFrame['Mean_Freq']
        privStd = dataFrame['StdDev_Freq']
        lengthTrueF = len(trueFreq)

        fiftyPerc = int(0.5 * lengthTrueF)
        ninetyPerc = int(0.1 * lengthTrueF)
        ninetyFivePerc = int(0.05 * lengthTrueF)

        row = [i]
        row.append((1.0 * trueFreq[fiftyPerc]) / config.config.n)
        row.append((1.0* privteFreq[fiftyPerc])/config.config.n)
        row.append((1.0 * privStd[fiftyPerc]/config.config.n) if not math.isnan(privStd[fiftyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privStd[ninetyPerc]) / config.config.n if not math.isnan(privStd[ninetyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyFivePerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyFivePerc]) /config.config.n)
        row.append((1.0 * privStd[ninetyFivePerc])/config.config.n if not math.isnan(privStd[ninetyFivePerc]) else 0.0)
        resutDataFramne.loc[y] = row

    drawGroupedBarGraph(resutDataFramne, displayLabels, 'ExperimentVaryingN', yLabel='Estimated frequency', title='Estimated frequency versus n')

# Experiment 2: Run with varying values of epsilon
def runFrequencyEstimationWithVaryingEps():
    config.config.reinitializeParameters()
    nChoices = [0.1,1.0,5.0,10.0]
    displayLabels = ['50pctle','90pctle','95pctle']
    columns = ['epsilon','50pctle-True','50pctle-Priv','50pctle-Priv-Std','90pctle-True','90pctle-Priv','90pctle-Priv-Std','95pctle-True','95pctle-Priv','95pctle-Priv-Std']
    resutDataFramne = pd.DataFrame(columns = columns)
    for y,i in enumerate(nChoices):
        print 'Running expt for epsilon=', i
        filePrefix = str(uuid.uuid4()) + '_'

        [configFileName, reportFileName] = runWithPowerLawDist(filePrefix=filePrefix, epsilon=i)
        dataFrame = pd.read_csv(config.config.dataPath + filePrefix + reportFileName)

        trueFreq = dataFrame['trueFrequency']
        privteFreq = dataFrame['Mean_Freq']
        privStd = dataFrame['StdDev_Freq']
        lengthTrueF = len(trueFreq)

        fiftyPerc = int(0.5 * lengthTrueF)
        ninetyPerc = int(0.1 * lengthTrueF)
        ninetyFivePerc = int(0.05 * lengthTrueF)

        row = [i]
        row.append((1.0 * trueFreq[fiftyPerc]) / config.config.n)
        row.append((1.0* privteFreq[fiftyPerc]) / config.config.n)
        row.append((1.0 * privStd[fiftyPerc]) / config.config.n if not math.isnan(privStd[fiftyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privStd[ninetyPerc])/ config.config.n if not math.isnan(privStd[ninetyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyFivePerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyFivePerc]) /config.config.n)
        row.append((1.0 * privStd[ninetyFivePerc]) / config.config.n if not math.isnan(privStd[ninetyFivePerc]) else 0.0)
        resutDataFramne.loc[y] = row
    
    print resutDataFramne
    rawGroupedBarGraph(resutDataFramne, displayLabels, 'ExperimentVaryingEps', yLabel='Estimated frequency', title='Estimated frequency versus epsilon')

# Experiment 3: Run with varying values of k
def runFrequencyEstimationWithVaryingW():
    config.config.reinitializeParameters()
    nChoices = [0.1,1.0,5.0,10.0]
    displayLabels = ['50pctle','90pctle','95pctle']
    columns = ['w','50pctle-True','50pctle-Priv','50pctle-Priv-Std','90pctle-True','90pctle-Priv','90pctle-Priv-Std','95pctle-True','95pctle-Priv','95pctle-Priv-Std']
    resutDataFramne = pd.DataFrame(columns = columns)
    for y,i in enumerate(nChoices):

        print 'Running expt for w=', i
        filePrefix = str(uuid.uuid4()) + '_'

        [configFileName, reportFileName] = runWithPowerLawDist(filePrefix=filePrefix, w = config.config.nearestPowerOfTwoCeil(i * math.sqrt(config.config.n)))
        dataFrame = pd.read_csv(config.config.dataPath + filePrefix + reportFileName)

        trueFreq = dataFrame['trueFrequency']
        privteFreq = dataFrame['Mean_Freq']
        privStd = dataFrame['StdDev_Freq']
        lengthTrueF = len(trueFreq)

        fiftyPerc = int(0.5 * lengthTrueF)
        ninetyPerc = int(0.1 * lengthTrueF)
        ninetyFivePerc = int(0.05 * lengthTrueF)

        row = [i]
        row.append((1.0 * trueFreq[fiftyPerc]) / config.config.n)
        row.append((1.0* privteFreq[fiftyPerc])/config.config.n)
        row.append((1.0 * privStd[fiftyPerc]/config.config.n) if not math.isnan(privStd[fiftyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privStd[ninetyPerc]) / config.config.n if not math.isnan(privStd[ninetyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyFivePerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyFivePerc]) /config.config.n)
        row.append((1.0 * privStd[ninetyFivePerc])/config.config.n if not math.isnan(privStd[ninetyFivePerc]) else 0.0)
        resutDataFramne.loc[y] = row

    drawGroupedBarGraph(resutDataFramne, displayLabels, 'ExperimentVaryingW', yLabel='Estimated frequency', title='Estimated frequency versus sketch-width x sqrt(n)')

# Experiment 4: Run with varying values of epsilon
def runFrequencyEstimationWithVaryingEpsNLTK():
    config.config.reinitializeParameters()
    nChoices = [0.1, 1.0, 2.0, 5.0, 10.0]
    displayLabels = ['Rank 1','Rank 10','Rank 100']
    columns = ['epsilon','Rank 1-True','Rank 1-Priv','Rank 1-Priv-Std','Rank 10-True','Rank 10-Priv','Rank 10-Priv-Std','Rank 100-True','Rank 100-Priv','Rank 100-Priv-Std']
    resutDataFramne = pd.DataFrame(columns = columns)
    for y,i in enumerate(nChoices):
        print 'Running expt for epsilon=', i
        filePrefix = str(uuid.uuid4()) + '_'

        [configFileName, reportFileName] = runWithBrownCorpursNLTK(filePrefix=filePrefix, epsilon=i)
        dataFrame = pd.read_csv(config.config.dataPath + filePrefix + reportFileName)

        trueFreq = dataFrame['trueFrequency']
        privteFreq = dataFrame['Mean_Freq']
        privStd = dataFrame['StdDev_Freq']
        lengthTrueF = len(trueFreq)

        fiftyPerc = 0 # int(0.5 * lengthTrueF)
        ninetyPerc = 9 # int(0.1 * lengthTrueF)
        ninetyFivePerc = 99 # int(0.05 * lengthTrueF)

        row = [i]
        row.append((1.0 * trueFreq[fiftyPerc]) / config.config.n)
        row.append((1.0* privteFreq[fiftyPerc]) / config.config.n)
        row.append((1.0 * privStd[fiftyPerc]) / config.config.n if not math.isnan(privStd[fiftyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyPerc]) / config.config.n)
        row.append((1.0 * privStd[ninetyPerc])/ config.config.n if not math.isnan(privStd[ninetyPerc]) else 0.0)

        row.append((1.0 * trueFreq[ninetyFivePerc]) / config.config.n)
        row.append((1.0 * privteFreq[ninetyFivePerc]) /config.config.n)
        row.append((1.0 * privStd[ninetyFivePerc]) / config.config.n if not math.isnan(privStd[ninetyFivePerc]) else 0.0)
        resutDataFramne.loc[y] = row
    
    print resutDataFramne
    drawGroupedBarGraph(resutDataFramne, displayLabels, 'BrownExperimentVaryingEps', yLabel='Estimated frequency', title='Estimated frequency versus epsilon')

# Experiment 5: Compare with Rappor
def getDetailsForRappor():
    path = '../../dataLoc/'
    dfTrue = pd.read_csv(path + 'RapporTrue.csv')
    dataTF = {}

    for i, row in enumerate(dfTrue.iterrows()):
        if row[1]['value'] not in dataTF.keys():
            dataTF[row[1]['value']] = 1
        else:
            dataTF[row[1]['value']] += 1

    tokens_with_count = sorted([(x, dataTF[x], 0.0, 0.0) for x in dataTF.keys()], key=lambda i: i[1], reverse=True)
    dfRappor = pd.read_csv(path + 'RapporPriv.csv')

    for i, row in enumerate(dfRappor.iterrows()):
        if row[1]['string'] not in dataTF.keys():
            continue
        for j in range(len(tokens_with_count)):
            [t1, t2, t3, t4] = tokens_with_count[j]
            if t1 == row[1]['string']:
                tokens_with_count[j] = (t1, t2, row[1]['estimate'], row[1]['std_error'])

    wordFrequency = pd.DataFrame(tokens_with_count, columns=['word', 'trueFrequency', 'RapporFrequency', 'RapporStd'])
    return wordFrequency

def compareWithRappor():
    config.config.reinitializeParameters()
    tableWithRapporRun = getDetailsForRappor()
    config.config.n = sum(tableWithRapporRun['trueFrequency'])
    config.config.epsilon = math.log(3.0, math.e)
    config.config.cEpsilon = (math.exp(config.config.epsilon) + 1) / (math.exp(config.config.epsilon) - 1)  # Scaling for de-biasing
    config.config.p = tableWithRapporRun.count()['word']

    wordFrequency = tableWithRapporRun[['word','trueFrequency']]
    ServerSide.runServerSide(wordFrequency, 'configRAPPOR.txt', 'corpusRappor.csv', filePrefix='')

    dataFrame = pd.read_csv(config.config.dataPath + 'corpusRappor.csv')

    newTableWithRappor = tableWithRapporRun[['word','trueFrequency']]
    newTableWithRappor = newTableWithRappor.merge(dataFrame[['word','Mean_Freq','StdDev_Freq']], how='inner', left_on='word', right_on='word')
    newTableWithRappor = newTableWithRappor.merge(tableWithRapporRun[['word', 'RapporFrequency', 'RapporStd']], how='inner', left_on='word', right_on='word')

    print newTableWithRappor

    numGroups = len(newTableWithRappor[[0]])
    pos = list(range(numGroups))
    width = 1.0 / (3 + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    my_colors = 'rgb'

    plt.bar([p for p in pos], np.array(newTableWithRappor[newTableWithRappor.columns[1]]) / (1.0 * config.config.n),width,
            color = my_colors[0])
    plt.bar([p+width for p in pos], np.array(newTableWithRappor[newTableWithRappor.columns[2]]) / (1.0 * config.config.n), width,
            color=my_colors[1]) #yerr = np.array(newTableWithRappor[newTableWithRappor.columns[3]]) / (1.0 * config.config.n),error_kw=dict(capsize=5))
    plt.bar([p + 2*width for p in pos], np.array(newTableWithRappor[newTableWithRappor.columns[4]]) / (1.0 * config.config.n), width,
            color=my_colors[2]) #yerr= np.array(newTableWithRappor[newTableWithRappor.columns[5]]) / (1.0 * config.config.n), error_kw=dict(capsize=5))

    ax.set_ylabel('Frequency estimate')
    ax.set_title('Comparison between Count-Sketch and RAPPOR')

    plt.xlim(min(pos) - width, max(pos) + 3.0 * width  + width)
    maxFreq = max(newTableWithRappor[newTableWithRappor.columns[1]]) / (1.0 * config.config.n)
    plt.ylim([0, 2.0 * maxFreq])

    xLeg = ['True Freq', 'Count_Sketch', 'RAPPOR']
    plt.legend(xLeg, loc='upper left')
    plt.grid()
    plt.savefig(config.config.dataPath + 'RAPPORComparison_frequency.pdf', bbox_inches='tight')


# runFrequencyEstimationWithVaryingN()
#runFrequencyEstimationWithVaryingEpsNLTK()
# runFrequencyEstimationWithVaryingW()
runWithPowerLawDist(wordDiscovery=True)

