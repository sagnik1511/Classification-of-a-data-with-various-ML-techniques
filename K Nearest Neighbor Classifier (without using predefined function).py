# libraries
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')




def k_n_n(data,predict,k=3):
    if len(data)>=k:
        warnings.earn('WARNING')
    distances=[]
    for group in data:
        for features in data[group]:
            eu_dist=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eu_dist,group])
    votes=[i[1] for i in sorted(distances) [:k]]
   # print(Counter(votes).most_common(1))
    vote_result=Counter(votes).most_common(1)[0][0]
        
    return vote_result
df=pd.read_csv(r'C:\Users\open\Downloads\breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['Id'],1,inplace=True)
# print(df.head())
full_data=df.astype(float).values.tolist()
# print(full_data[:5])
random.shuffle(full_data)
# print(end=" ")
# print(full_data[:5])
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])
correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        votes=k_n_n(train_set,data,k=5)
        print(votes)
        if group==votes:
            correct+=1
        total+=1
print(correct/total)