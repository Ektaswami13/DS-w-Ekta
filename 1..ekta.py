#ekta
#1 column dropped

import pandas as pd
data=pd.read_csv(r"D:\ASSIGNMENT\ekta\50_Startups.csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()





SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])

SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])

SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
data['State']=LB.fit_transform(data['State'])

print(data.corr()['Profit'])


X=data.iloc[:,0:3].values
Y=data.iloc[:,-1].values





from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)





from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
reg1=KNeighborsRegressor(n_neighbors=3)
reg2=SVR(kernel='poly',degree=20)
reg3= DecisionTreeRegressor(random_state=(0),max_depth=(4))
reg1.fit(X,Y)
reg2.fit(X,Y)
reg3.fit(X,Y)


y_pred_SVR=reg2.predict(X_test)
y_pred_KNN=reg1.predict(X_test)
y_pred_Decision=reg3.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(Y_test,y_pred_SVR)
print("Score by Support vector regressor")
print(score)
score=r2_score(Y_test,y_pred_KNN)
print("Score by KNN regressor ")
print(score)
score=r2_score(Y_test,y_pred_Decision)
print("score by DTR ")
print(score)





