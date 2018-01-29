# Generate data from RAPPOR
import pandas as pd
path = '../../dataLoc/'
dfTrue = pd.read_csv(path + 'RapporTrue.csv')
dataTF = {}

for i, row in enumerate(dfTrue.iterrows()):
    if row[1]['value'] not in dataTF.keys():
        dataTF[row[1]['value']] = 1
    else:
        dataTF[row[1]['value']] += 1

tokens_with_count = sorted([(x, dataTF[x],0.0,0.0) for x in dataTF.keys()], key=lambda i: i[1], reverse = True)
dfRappor = pd.read_csv(path + 'RapporPriv.csv')

for i, row in enumerate(dfRappor.iterrows()):
    if row[1]['string'] not in dataTF.keys():
        continue
    for j in range(len(tokens_with_count)):
        [t1, t2, t3, t4] = tokens_with_count[j]
        if t1 == row[1]['string']:
            tokens_with_count[j] = (t1, t2, row[1]['estimate'],row[1]['std_error'])

wordFrequency = pd.DataFrame(tokens_with_count, columns=['word','trueFrequency', 'RapporFrequency', "RapporStd"])
print sum(wordFrequency['trueFrequency'])
print wordFrequency.count()['word']
