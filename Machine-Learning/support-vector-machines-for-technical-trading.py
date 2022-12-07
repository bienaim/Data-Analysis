#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:40:13 2019

@author: Abien
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.preprocessing import StandardScaler

print('SUPPORT VECTOR MACHINES - USING WEEKLY PRICE AVERAGES AND STANDARD DEVIATIONS FROM YEAR 1,')
print('HOW WELL DOES THE CLASSIFIER FORECAST WEEKLY STOCK MOVEMENT FOR YEAR 2?')

#import file and create data frames
df = pd.read_csv("BTC-USD.csv")

df_2017_2018 = df[df.td_year.isin([2017,2018])]

first_day_df = df_2017_2018.groupby(['td_year','td_week_number']).first()
last_day_df = df_2017_2018.groupby(['td_year','td_week_number']).last()
avgWeeklyReturn_df = df_2017_2018.groupby(['td_year','td_week_number'])['return'].mean().reset_index(name="mean")
stdDevWeeklyReturn_df = df_2017_2018.groupby(['td_year','td_week_number'])['return'].std().reset_index(name="std_dev")

#create dictionaries to hold returns, averages, and standard deviations
weekly_return_dict = { }
valueSeries = pd.Series()
avgSeries = pd.Series()
stdDevSeries = pd.Series()

last_digit_dict = { }

#set labels based on weekly returns
def labels():

    df_length = (len(first_day_df))
    
    i=0
    while i < df_length:
        
            weekly_open = first_day_df.iloc[i,5]
            weekly_close = last_day_df.iloc[i,10]
            weekly_return = (weekly_close/weekly_open)-1
            #avgSeries.set_value(i, avgWeeklyReturn_df.iloc[i,2])
            avgSeries.at[i] = avgWeeklyReturn_df.iloc[i,2]

            #stdDevSeries.set_value(i, stdDevWeeklyReturn_df.iloc[i,2])
            stdDevSeries.at[i] = stdDevWeeklyReturn_df.iloc[i,2]

            
            if weekly_return > 0:
                #valueSeries.set_value(i,'green')
                valueSeries.at[i] = 'green'

                
            else:
                #valueSeries.set_value(i,'red')
                valueSeries.at[i] = 'red'

                    
            weekly_return_dict[i] = weekly_return
            i += 1
            
labels()

last_day_df = last_day_df.assign(label=valueSeries.values)
last_day_df['mean'] = avgSeries.values
last_day_df['std_dev'] = stdDevSeries.values

last_day_df = last_day_df[~last_day_df.trade_date.isin(['2017-12-31'])]

stock_feature_names = ['mean', 'std_dev']
data = pd.DataFrame(last_day_df, columns=['mean', 'std_dev', 'label'])
class_labels = ['red','green']
data = data[data['label'].isin(class_labels)]
# x variable


X = data[stock_feature_names].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform (X)
le = LabelEncoder()
Y = le.fit_transform(data['label'].values)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, shuffle=False)

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X,Y)


print('\nLINEAR SUPPORT VECTOR MACHINE ACCURACY')
print('The forecasted labels for year 2 are:\n' + str(prediction))
print('When using the Linear Support Vector Machine classifier, the accuracy for year 2 is ' + str(round(accuracy,4)) + '.')

cMatrix = confusion_matrix(Y_test, prediction)

# print the confusion matrix
print('\nLINEAR SUPPORT VECTOR MACHINE - CONFUSION MATRIX FOR YEAR 2')
print(cMatrix)

print('\nLINEAR SUPPORT VECTOR MACHINE - TRUE POSITIVE AND TRUE NEGATIVE RATES')
print('The true positive rate is ' + str(round(cMatrix[0,0]/(cMatrix[0,0]+cMatrix[0,1]),2)))
print('The true negative rate is ' + str(round(cMatrix[1,1]/(cMatrix[1,0]+cMatrix[1,1]),2)))

svm_accuracy = accuracy

svm_classifier = svm.SVC(kernel='rbf')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X,Y)

gaussian_svm_accuracy = accuracy

print('\n\nGAUSSIAN SUPPORT VECTOR MACHINE ACCURACY')
print('The forecasted labels for year 2 are:\n' + str(prediction))
print('When using the Gaussian Support Vector Machine classifier, the accuracy for year 2 is ' + str(round(accuracy,4)) + '.\n')

if svm_accuracy > gaussian_svm_accuracy:
    print('The Linear support vector machine model provides more accurate forecasts than the Gaussian support vector machine model.')
elif gaussian_svm_accuracy > svm_accuracy:
    print('The Gaussian support vector machine model provides more accurate forecasts than the Linear support vector machine model.')
else:
    print('The two have the same accuracy')

svm_classifier = svm.SVC(kernel='poly', degree=2)
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X,Y)

polynomial_svm_accuracy = accuracy


print('\n\nPOLYNOMIAL SUPPORT VECTOR MACHINE ACCURACY')
print('The forecasted labels for year 2 are:\n' + str(prediction))
print('When using the Polynomial Support Vector Machine classifier, the accuracy for year 2 is ' + str(round(accuracy,4)) + '.\n')

if svm_accuracy > polynomial_svm_accuracy:
    print('The Linear support vector machine model provides more accurate forecasts than the Polynomial support vector machine model.')
elif polynomial_svm_accuracy > svm_accuracy:
   print('The Polynomial support vector machine model provides more accurate forecasts than the Linear support vector machine model.')
else:
    print('The two have the same accuracy')



svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X,Y)


last_day_df_2018 = last_day_df[last_day_df['trade_date'].str.contains('2018')]

def strategy1():

    USD_value = 100
    BTC_value = 0
    invested = False
    i=0
        
    while i < (len(last_day_df_2018)-1):
              
                        if invested == False:
                            shares = (USD_value + BTC_value)/(last_day_df_2018.iloc[i,10])
                        else:
                            BTC_value += float(last_day_df_2018.iloc[i,10])*float(shares)-float(BTC_value)                        
                        
                        if prediction[i+1] == 0:
                            if invested == False:
                                BTC_value = float(last_day_df_2018.iloc[i,10]) * float(shares)
                                USD_value = USD_value - BTC_value
                                invested = True
                        elif prediction[i+1] == 1:
                            if invested == True:
                                USD_value +=  BTC_value
                                BTC_value = 0
                                invested = False
               
                        i += 1
                        

    print('\n\vTRADING STRATEGY 1')                    
    print('If you start with $100 and buy or sell using the forecast labels,')
    print('during the last week, the value will be $' + str(round(USD_value,2)) + ' and $' + str(round(BTC_value))+' BTC\n')

#use buy and hold strategy to buy with $100 and sell on the last trading day
def strategy_buy_hold():

                    amount = 100*((round(float(last_day_df_2018.iloc[len(last_day_df_2018)-1,10])/float(last_day_df_2018.iloc[0,10]),2))-1)
        
                    print('\nBUY AND HOLD TRADING STRATEGY')
                    print('A buy and hold strategy would return $' + str(amount) + '.\n' )

strategy1()
strategy_buy_hold()