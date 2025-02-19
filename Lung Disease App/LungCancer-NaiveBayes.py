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
 
INPUT = [0,59,1,2,2,2,2,2,2,2,1,2,2,2,1]

test_predictions = alg.predict([INPUT])
print(test_predictions)
if test_predictions == 1:
    print("Patient Having Disease")
else:
    print("No Disease")
 