import pandas as pd
data=pd.read_csv(r"E:\heart.csv")
data.isna().sum()
x=data.iloc[0:,0:13].values
y=data.iloc[0:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=30)

from sklearn.ensemble import RandomForestClassifier as r

from sklearn.metrics import accuracy_score
n=[]
scr=[]
for i in range(3,20):
    model=r(n_estimators=i,random_state=2)
    model.fit(x_train,y_train)
    Y_pred=model.predict(x_test)
    score=accuracy_score(Y_pred,y_test)
    n.append(i)
    scr.append(score)
import matplotlib.pyplot as plt
plt.plot(n,scr)
plt.xlabel("no_estimator")
plt.ylabel("Accuracy")
plt.title("no_estimator vs accuracy for Randomforest\nHeart Disease Data")
plt.show()   