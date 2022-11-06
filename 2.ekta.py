#ekta
#2 column dropped
import pandas as pd
data=pd.read_csv(r"D:\ASSIGNMENT\ekta\50_Startups.csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()





SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])





SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])




SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())





print(data.corr()['Profit'])
data=data.drop('State',axis=1)






X=data.iloc[:,0:3:1].values
Y=data.iloc[:,-1].values
print(X)





from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)





from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)





from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor(n_neighbors=3)
reg1=SVR(kernel='poly',degree=20)
reg2= DecisionTreeRegressor(random_state=(0),max_depth=(4))
reg.fit(X,Y)
reg1.fit(X,Y)
reg2.fit(X,Y)



y_pred_Decision=reg2.predict(X_test)
y_pred_SVR=reg1.predict(X_test)
y_pred_KNN=reg.predict(X_test)


from sklearn.metrics import r2_score
score1=r2_score(Y_test,y_pred_SVR)
print("Score by SVR : ",score1)


score2=r2_score(Y_test,y_pred_KNN)
print("Score by KNN Regressor : ",score2)


score3=r2_score(Y_test,y_pred_Decision)
print("Score by DTR : ",score3)


