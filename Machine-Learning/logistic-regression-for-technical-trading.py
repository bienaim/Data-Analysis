#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:40:13 2019

@author: Abien
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator

print('LOGISTIC REGRESSION - USING WEEKLY PRICE AVERAGES AND STANDARD DEVIATIONS FROM YEAR 1,')
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

def labels():
    #set labels

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
#print('\nWEEKLY_RETURN_LABELS')  

#print(valueSeries) 
last_day_df = last_day_df.assign(label=valueSeries.values)
last_day_df['mean'] = avgSeries.values
last_day_df['std_dev'] = stdDevSeries.values


last_day_df = last_day_df[~last_day_df.trade_date.isin(['2017-12-31'])]



#logisticRegression
stock_feature_names = ['mean', 'std_dev']
data = pd.DataFrame(last_day_df, columns=['mean', 'std_dev', 'label'])
# x variable
X = data[stock_feature_names].values

# scale data to normalize features
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X)
le = LabelEncoder ()
# y variable
Y = le.fit_transform(data['label']) 

# creating training and test splits where training set size is test_size
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5, shuffle=False)

log_reg_classifier = LogisticRegression() 
log_reg_classifier.fit(X_train ,Y_train)

prediction = log_reg_classifier.predict(X_test) 
accuracy = np.mean(prediction == Y_test)

print('\nLOGISTIC REGRESSION EQUATION FOR YEAR 2')
equation = str('y = ' + str(round(log_reg_classifier.intercept_[0],4)) + ' + ' 
+ str(round(log_reg_classifier.coef_[0][0],4)) + ' X1 + '
+ str(round(log_reg_classifier.coef_[0][1],4)) + ' X2')

print(equation)

print('\nSigmoid function')
print('z(y) = 1/[1 + e**(-(' + equation + '))]')

print()


print('\nLOGISTIC REGRESSION ACCURACY FOR YEAR 2')
print('The predicted labels for year 2 are:\n' + str(prediction))
print('The accuracy for 2018 is ' + str(round(accuracy,4)) + '.')

cMatrix = confusion_matrix(Y_test, prediction)
    # print the confusion matrix
print('\nLOGISTIC REGRESSION CONFUSION MATRIX FOR 2018')
print(cMatrix)

print('\nThe true positive rate is ' + str(round(cMatrix[0,0]/(cMatrix[0,0]+cMatrix[0,1]),2)))
print('The true negative rate is ' + str(round(cMatrix[1,1]/(cMatrix[1,0]+cMatrix[1,1]),2)))


    

last_day_df_2018 = last_day_df[last_day_df['trade_date'].str.contains('2018')]

def strategy1():
    #use labels to start with $20,000, buy when green, sell when red

    USD_value = 20000.0
    BTC_value = 0
    invested = False
    i=0
        
    while i < (len(last_day_df_2018)-1):
              
                        if invested == False:
                            shares = int((USD_value + BTC_value)/float(last_day_df_2018.iloc[i,10]))
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
                        

                
    print('If you start with $20,000 and trade weekly based on the forecast labels,')
    print('during the last week, the value will be $' + str(round(USD_value,2)) + ' and BTC ' + str(round(BTC_value))+'\n')

def strategy_buy_hold():
    #use buy and hold strategy to buy with $1,000 and sell on the last trading day

                    amount = round(float(last_day_df_2018.iloc[len(last_day_df_2018)-1,10]) - float(last_day_df_2018.iloc[0,10]),2)
        
                    print('\nBUY AND HOLD TRADING STRATEGY')
                    print('A buy and hold strategy would return $' + str(amount) + '.\n' )

print('\nLOGISTIC REGRESSION TRADING STRATEGY')    
strategy1()
strategy_buy_hold()