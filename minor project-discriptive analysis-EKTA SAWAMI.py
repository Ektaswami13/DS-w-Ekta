#descriptive analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os
#os.getcwd()
import os
df=pd.read_excel("C:\Users\Acer\OneDrive\Documents\ekta\Scores_of_students (1).xlsx")
df1=pd.DataFrame(df)

ind=df1.index
df2=df1[['Mathematics','Physics','Chemistry','Total']]
#print(df1)
def maxim(sub):
    if(sub==1):
        con=df1['Mathematics']==df1['Mathematics'].max()
        nm=ind[con].tolist()
        print("Name of student who got maximum marks in mathematics:")
        for i in nm:
            print(df['Name'][i])
        p=df1["Mathematics"].idxmax()
        print("Maximum marks in math:",df1["Mathematics"].max())
    elif(sub==2):
        con=df1['Chemistry']==df1['Chemistry'].max()
        nm=ind[con].tolist()
        print("Name of student who got maximum marks in chemistry")
        for i in nm:
            print(df['Name'][i])
        p=df1["Chemistry"].idxmax()
        print("Maximum marks in math:",df1["Chemistry"].max())
    elif(sub==3):
        con=df1['Physics']==df1['Physics'].max()
        nm=ind[con].tolist()
        print("Name of student who got maximum marks in physics:")
        for i in nm:
            print(df['Name'][i])
        p=df1["Physics"].idxmax()
        print("Maximum marks in math:",df1["Physics"].max())
    return 
def mode():
    m=statistics.mode((df1["Mathematics"]))
    q=statistics.mode((df1["Physics"]))
    l=statistics.mode((df1["Chemistry"]))
    x=[m,q,l]
    y=['math','physics','chemistry']
    print("mode of mathematics",m)
    print("mode of Physics:",q)
    print("mode of Chemistry:",l)
    plt.bar(y,x)
    plt.show()
    return
def stdd():
    k=np.std(df1["Physics"])
    print("Standard Daviation of Physics:",k)
    return
def mxov():
    j=df1["Total"].idxmax()
    print("highest overall marks:",df1["Total"].max(),"obtained by",df['Name'][j],"from",df["CityTown"][j])
    return
def totalbng():
    tg=0
    tb=0
    for i in df1['Gender']:
        if(i=='F'):
            tg=tg+1
        else:
            tb=tb+1
    print("Total number of Girls:",tg)
    print("TOtal number of Boys:",tb)
    v=[tg,tb]
    labels=["Girls","Boys"]
    plt.pie(v,labels=labels)
    plt.show()
def avrgbng():
    con1=df1['Gender']=='F'
    con2=df1['Gender']=='M'
    mgt=df1['Total'][con1].mean()
    mbt=df1['Total'][con2].mean()
    print("Average total marks of girls:",mgt)
    print("Average total marks of boys:",mbt)
    v=[mgt,mbt]
    lab=["Girls","Boys"]
    plt.pie(v,labels=lab)
   
    plt.show()
    #fig=plt.figure(figsize=(10))
    
    return
def subavr():
    mav=df1['Mathematics'].mean()
    pav=df1['Physics'].mean()
    cmv=df1['Chemistry'].mean()
    print("Average marks in Math:",mav)
    print("Average marks in Physics",pav)
    print("Average marks in Chemistry",cmv)

print('''1.show Maximum marks in in particular subject with name
         2.Show overall highest marks in dataset
         3.Mode of Mode of mathematics
         4.Satandard Deviation for physics
         5.Show total number of boys and Girls
         6.Show average total marks of boys and girls
         7. Average marks by subject
         ''')
o=int(input("Select option:"))
if(o==1):
    print('''1.Maximum marks in math
             2.maximum marks in chemistry
             3.maximum marks in Physics
    choose option''')
    ch=int(input("choose option"))
    maxim(ch)
elif(o==2):
    mxov()
elif(o==3):
    mode()
elif(o==4):
    stdd()
elif(o==5):
    totalbng()
elif(o==6):
    avrgbng()
elif(o==7):
    subavr()
    
else:
    print("Invalid input")
plt.figure(figsize=[30,10])
plt.bar(df1['Name'],df2['Total'])

plt.xticks(rotation=1)