import numpy as np
from sklearn import preprocessing,neighbors
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'C:\Users\open\Downloads\breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['Id'],1,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)

print(accuracy)

exp_measures=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
exp_measures=exp_measures.reshape(len(exp_measures),-1)
pred=clf.predict(exp_measures)
print(pred)
