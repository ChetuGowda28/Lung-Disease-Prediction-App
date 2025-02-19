
import tkinter as tk

from tkinter import *

window = tk.Tk()

window.title('LungDiseaseApp')

window.configure(background = 'green')

window.geometry('1250x700')

window.grid_rowconfigure(0, weight = 1)

window.grid_columnconfigure(0, weight = 1)


lb = tk.Label(window, text = ' LUNG DISEASE APPLICATION', bg = 'black', fg = 'white', font = ('times', 30, 'italic bold underline'))
lb.place(x = 350, y = 20)


#ALCOHOL CONSUMING,COUGHING,SHORTNESS OF BREATH,SWALLOWING DIFFICULTY,CHEST PAIN


age = tk.Label(window,text = "Enter the Age in years : ", bg = 'blue', fg ='white')

age.place(x =100, y = 100)


gender = tk.Label(window,text = "Enter Gender 1(Male) 0 (Female): ", bg = 'blue', fg ='white')

gender.place(x =100, y = 150)


cp = tk.Label(window,text = "Enter SMOKING: ", bg = 'blue', fg ='white')

cp.place(x =100, y = 200)


trestbps = tk.Label(window,text = "Enter YELLOW_FINGERS: ", bg = 'blue', fg ='white')

trestbps.place(x =100, y = 250)


chol = tk.Label(window,text = "Enter ANXIETY: ", bg = 'blue', fg ='white')

chol.place(x =100, y = 300)


fbs = tk.Label(window,text = "Enter PEER_PRESSURE: ", bg = 'blue', fg ='white')

fbs.place(x =100, y = 350)



restecg = tk.Label(window,text = "Enter CHRONIC DISEASE: ", bg = 'blue', fg ='white')

restecg.place(x =100, y = 400)


thalach = tk.Label(window,text = "Enter FATIGUE: ", bg = 'blue', fg ='white')

thalach.place(x =100, y = 450)



exang = tk.Label(window,text = "Enter ALLERGY: ", bg = 'blue', fg ='white')

exang.place(x =100, y = 500)



oldpeak = tk.Label(window,text = "Enter WHEEZING: ", bg = 'blue', fg ='white')

oldpeak.place(x =100, y = 550)


slope = tk.Label(window,text = "Enter ALCOHOL CONSUMING: ", bg = 'blue', fg ='white')

slope.place(x =100, y = 600)


ca = tk.Label(window,text = "Enter COUGHING: ", bg = 'blue', fg ='white')

ca.place(x =700, y = 100)


thal = tk.Label(window,text = "Enter SHORTNESS OF BREATH: ", bg = 'blue', fg ='white')

thal.place(x =700, y = 150)


swal = tk.Label(window,text = "Enter SWALLOWING DIFFICULTY: ", bg = 'blue', fg ='white')

swal.place(x =700, y = 200)

chestp = tk.Label(window,text = "Enter CHEST PAIN: ", bg = 'blue', fg ='white')

chestp.place(x =700, y = 250)



patage = StringVar()
ageEntry = tk.Entry(window, textvariable = patage)
ageEntry.place(x = 300, y = 100)

patgender = StringVar()
gEntry = tk.Entry(window, textvariable = patgender)
gEntry.place(x = 300, y = 150)

patcp = StringVar()
cpEntry = tk.Entry(window, textvariable = patcp)
cpEntry.place(x = 300, y = 200)

pattrest = StringVar()
trestEntry = tk.Entry(window, textvariable = pattrest)
trestEntry.place(x = 300, y = 250)

patchol = StringVar()
cholEntry = tk.Entry(window, textvariable = patchol)
cholEntry.place(x = 300, y = 300)


patfbs = StringVar()
fbsEntry = tk.Entry(window, textvariable = patfbs)
fbsEntry.place(x = 300, y = 350)


patrest = StringVar()
restEntry = tk.Entry(window, textvariable = patrest)
restEntry.place(x = 300, y = 400)


patthalach = StringVar()
thalachEntry = tk.Entry(window, textvariable = patthalach)
thalachEntry.place(x = 300, y = 450)


patexang = StringVar()
exangEntry = tk.Entry(window, textvariable = patexang)
exangEntry.place(x = 300, y = 500)


patoldpeak = StringVar()
oldpeakEntry = tk.Entry(window, textvariable = patoldpeak)
oldpeakEntry.place(x = 300, y = 550)


patslope = StringVar()
slopeEntry = tk.Entry(window, textvariable = patslope)
slopeEntry.place(x = 300, y = 600)



patca = StringVar()
caEntry = tk.Entry(window, textvariable = patca)
caEntry.place(x = 900, y = 100)


patthal = StringVar()
thalEntry = tk.Entry(window, textvariable = patthal)
thalEntry.place(x = 900, y = 150)

patswal = StringVar()
swalEntry = tk.Entry(window, textvariable = patswal)
swalEntry.place(x = 900, y = 200)


patchestp = StringVar()
chestpEntry = tk.Entry(window, textvariable = patchestp)
chestpEntry.place(x = 900, y = 250)


lbout = tk.Label(window,text = "Predicted Ouput is: ", bg = 'blue', fg ='white')

lbout.place(x =900, y = 470)



patout = StringVar()
outEntry = tk.Entry(window, textvariable = patout)
outEntry.place(x = 900, y = 500)


def pred():
    import pandas
    import os
    import numpy as np
    from sklearn.metrics import accuracy_score 
    #from sklearn.cross_validation import KFold
    #from sklearn import cross_validation
    from sklearn.naive_bayes import GaussianNB
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()

    lung = pandas.read_csv("lung_cancer.csv")

    lung.loc[lung["GENDER"]=="M","GENDER"]=1
    lung.loc[lung["GENDER"]=="F","GENDER"]=0
    lung.loc[lung["LUNG_CANCER"]=="YES","LUNG_CANCER"]=1
    lung.loc[lung["LUNG_CANCER"]=="NO","LUNG_CANCER"]=0

    print(lung.describe())

    predictors=["GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE","FATIGUE","ALLERGY ","WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN",]

    alg=GaussianNB()

    predictions = []
    train_predictors = (lung[predictors].iloc[:,:])
    print(train_predictors)
    train_target = lung["LUNG_CANCER"].iloc[:]

    label = le.fit_transform(train_target)
    print(label)
    alg.fit(train_predictors, label)
    l=len(lung.index)

    # Please enter the input attribute below
    ''' 
    INPUT = [0,59,1,2,2,2,2,2,2,2,1,2,2,2,1]

    test_predictions = alg.predict([INPUT])
    print(test_predictions)
    if test_predictions == 1:
        print("Patient Having Disease")
    else:
        print("No Disease")
     
    '''
    
    x1 = patage.get()
    
    x2 = patgender.get()
    
    x3 = patcp.get()
    
    x4 = pattrest.get()
    
    x5 = patchol.get()
    
    x6 = patfbs.get()
    
    x7 = patrest.get()
    
    x8 = patthalach.get()
    
    x9 = patexang.get()
    
    x10 = patoldpeak.get()
    
    x11 = patslope.get()
    
    x12 = patca.get()
    
    x13 = patthal.get()
    
    x14 = patswal.get()
    
    x15 = patchestp.get()
    
    inp = [int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8),int(x9),int(x10),int(x11),int(x12),int(x13), int(x14), int(x15)]
    
    print(inp)
    
    
    ypred = alg.predict([inp])
    
    
    print(ypred)
    
    if ypred[0] == 0:
        patout.set('No Lung Cancer')
        print('No Lung Cancer present for input patient')
    else:
        patout.set('Lung Cancer')
        print('Lung Cancer present for input patient')
    



but = tk.Button(window, text = 'Predict', command = pred, bg = 'red', fg = 'white', width = 20, height =1)
but.place(x = 800, y = 400)

window.mainloop()