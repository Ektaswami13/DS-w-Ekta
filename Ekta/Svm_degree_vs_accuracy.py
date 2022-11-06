import pandas as pd
data=pd.read_csv(r"E:\heart.csv")
data.isna().sum()
x=data.iloc[0:,0:13].values
y=data.iloc[0:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
from sklearn.svm import SVC
model=SVC(kernel='linear',random_state=1)
model.fit(x_train,y_train)
pred=model.predict(x_test)
#print(pred)
from sklearn.metrics import accuracy_score
score=accuracy_score(pred,y_test)

d=[]
sc=[]
from sklearn.svm import SVC
for i in range(1,6):
    model=SVC(kernel='poly',degree=i,random_state=1)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    #cm=confusion_matrix(Y_pred,Y_test)
    score=accuracy_score(y_pred,y_test)
    d.append(i)
    sc.append(score)
import matplotlib.pyplot as plt
plt.plot(d,sc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.xlabel("Value degree")
plt.ylabel("Accuracy")
plt.title("degree vs accuracy for svm\nHeart Disease data")
plt.show()