#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:55:21 2019

@author: Abien
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
#df_RandomForest = pd.read_csv("df_classifier_data_2016.csv").fillna(0)

print('FORECASTING AN ANNUAL INCREASE OR DECREASE IN TOURISM RECEIPTS USING THE RANDOM FOREST CLASSIFIER')

feature_names = ['New businesses registered (number)', 'Portfolio investment, bonds (PPG + PNG) (NFL, current US$)',
                 'Listed domestic companies, total','Air transport, passengers carried']
data = pd.read_csv("df_classifier_data.csv", names=['New businesses registered (number)', 'Portfolio investment, bonds (PPG + PNG) (NFL, current US$)',
                 'Listed domestic companies, total','Air transport, passengers carried', 
                 'Next Year International tourism, receipts (current US$) category']).fillna(0)
class_labels = ['red', 'green']
data = data[data['Next Year International tourism, receipts (current US$) category'].isin(class_labels)]
X = data[feature_names].values
le = LabelEncoder ()
Y = le.fit_transform(data['Next Year International tourism, receipts (current US$) category'].values)
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y, test_size=0.5,random_state=3)

#Function for random forest classifier with a given n and depth,d
def RandomForest(n,d):
    model = RandomForestClassifier(n_estimators=10, max_depth=5, criterion='entropy')
    
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    error_rate = np.mean(prediction != Y_test)
    importances = model.feature_importances_
    return error_rate, importances

accuracySeries = pd.Series()
nSeries= pd.Series()
dSeries= pd.Series()
featureSeries = pd.Series()

i=0

print('\n\nRANDOM FOREST CLASSIFIER - FORECASTING USING THE FOLLOWING FEATURES:')
print('New Businesses, Bond Investments, Listed Domestic Companies, Air Transport\n\n')
#run the Random Forest classifier with up to 10 trees, and depth ranging from 1 to 5
for n in range (1,11):
    for d in range(1, 6):
        
        #nSeries.set_value(i,n)
        nSeries.at[i] = n
        #dSeries.set_value(i,d)
        dSeries.at[i] = d
        
        error_rate, importances = RandomForest(n,d)
        accuracy = 1 - error_rate
        #accuracySeries.set_value(i,accuracy)
        accuracySeries.at[i] = accuracy
        # featureSeries.set_value(i, importances)
        featureSeries.at[i] = importances
                
        i+=1
    print('Added ' + str(n) + ' trees with depths ranging from 1 to 5')


#combine the series into a dataframe        
df_accuracy = pd.concat([nSeries, dSeries, accuracySeries, featureSeries],axis=1)

#find the highest accuracy
optimalParameters = df_accuracy[2].idxmax()
bestAccuracy = round(df_accuracy.iloc[optimalParameters][2],4)
optimalTrees = int(df_accuracy.iloc[optimalParameters][0])
optimalDepth = int(df_accuracy.iloc[optimalParameters][1])
importances = df_accuracy.iloc[optimalParameters][3]



print('\n\nRANDOM FOREST CLASSIFIER - WHAT IS THE OPTIMAL NUMBER OF TREES AND DEPTH TO USE?')
print('The best acheived accuracy was ' + str(bestAccuracy))
print('This was achieved with ' + str(optimalTrees) + ' trees and a depth of ' + str(optimalDepth) +'.')

#n_estimators is the number of trees in the forest
#max_depth The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure 
#or until all leaves contain less than min_samples_split samples.

model = RandomForestClassifier(n_estimators=optimalTrees, max_depth=optimalDepth, criterion='entropy')
    
#model.fit(X_train, Y_train)
#prediction = model.predict(X_test)
#error_rate = np.mean(prediction != Y_test)

#plot importance of each feature based on average decrease in impurity for each feature/split in the forest
indices = np.argsort(importances)
features = data.columns[0:4]

plt.figure(1)
plt.title('FEATURE IMPORTANCE')
plt.barh(range(len(indices)), importances[indices], color='cornflowerblue', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

print('\n\nRANDOM FOREST CLASSIFIER - HOW IMPORTANT IS EACH FEATURE')
print('FOR FORECASTING AN INCREASE OR DECREASE IN TOURISM?')

print('\nIt is important to note that although a feature may have high')
print('relative importance in the model, it doesn\'t necessarily')
print('cause the increases or decreases in tourism.')

print('\nA plot has been created showing relative feature importances.')

print('\n\nRANDOM FOREST CLASSIFIER')
print('When using New Businesses, Bond Investments, Listed Domestic Companies, and Air Transport')
print('as features, the random forest classifier can be used to forecast increases or decreases')
print('in next year\'s tourism with about ' + str(round(bestAccuracy,2)*100) + '% accuracy')


