#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 00:21:08 2019

@author: Abien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('BOLLINGER BANDS - If we want to buy a stock when its price is below its moving average')
print('and sell it when its price is above its moving average:')
print('\nhow many days (W) should we use for the moving average?')
print('\nHow far above and below the average should the price (k) be before we buy or sell?')
print('\nWhat combination of days, W and volatility standard deviation, k provides the best returns in each year?\n\n')

#import file and create data frames
df = pd.read_csv("BTC-USD.csv")
#data frames for year 1 and year 2 profit data
df_windowProfit2017 = pd.DataFrame(columns=['window','average_trade_profit','longTransactionCount','shortTransactionCount',
                                            'avg_long_profit','avg_short_profit','avg_days_long', 'avg_days_short'])
df_windowProfit2018 = pd.DataFrame(columns=['window','average_trade_profit','longTransactionCount','shortTransactionCount',
                                            'avg_long_profit','avg_short_profit','avg_days_long', 'avg_days_short'])
minWindow = 5
maxWindow = 8
windowSequence = list(range(minWindow,maxWindow+1))
standardDeviationSequence = list([0.5,1,1.5,2,2.5])

new_row = None

#function to return lower band, upper band, and signal given W, k, and year
def bollinger(windowsize, standardDeviation, year1):
    df_year = df[df.td_year.isin([year1])]
    upperBollingerSeries = pd.Series(dtype="float64")
    lowerBollingerSeries = pd.Series(dtype="float64")
    signalSeries = pd.Series(dtype="int64")
    
    W = windowsize
    k = standardDeviation
    df_length = (len(df_year))
    i=0
    upperBollingerSeries.append(pd.Series(np.arange(W)))
    lowerBollingerSeries.append(pd.Series(np.arange(W)))

    
    while i < df_length:
        if i < W:            
            #upperBollingerSeries.set_value(i,0)
            upperBollingerSeries.at[i] = 0

            #lowerBollingerSeries.set_value(i,0)
            lowerBollingerSeries.at[i] = 0
            
            #signalSeries.set_value(i,0)
            signalSeries.at[i] = 0

            
            i += 1
        else:
            W_values = np.array(df_year.iloc[i-W:i,12].values)
            average = np.average(W_values)
            std_dev = np.std(W_values)
            
            upperBand = average + (k*std_dev)
            lowerBand = average - (k*std_dev)
            
            #upperBollingerSeries.set_value(i, upperBand)
            upperBollingerSeries.at[i] = upperBand

            #lowerBollingerSeries.set_value(i, lowerBand)
            lowerBollingerSeries.at[i] = lowerBand

            
            if lowerBollingerSeries[i]>df_year.iloc[i,12]:
                #signalSeries.set_value(i,1)
                signalSeries.at[i] = 1

                
            elif upperBollingerSeries[i]<df_year.iloc[i,12]:
                #signalSeries.set_value(i,-1)
                signalSeries.at[i] = -1

            else:
                #signalSeries.set_value(i,0)
                signalSeries.at[i] = 0

            i+= 1
    
    window_data = pd.concat([lowerBollingerSeries, upperBollingerSeries, signalSeries],axis=1)
    window_data = window_data.rename(columns={0:str(W)+'_'+str(k)+'_lowerBand', 1:str(W)+'_'+str(k)+'_upperBand', 2:str(W)+'_'+str(k)+'_signal'})
    return window_data

#function to return profit/loss data and long/short position statistics given W, k, and year
def strategy1(window,standardDeviation,year1):
    W = window
    k = standardDeviation
    if year1 == 2017:
        df_window = pd.DataFrame(df_2017, columns = ['trade_date','open','adj_close', str(W)+'_'+str(k)+'_lowerBand', str(W)+'_'+str(k)+'_upperBand', str(W)+'_'+str(k)+'_signal'])
    elif year1 == 2018:
        df_window = pd.DataFrame(df_2018, columns = ['trade_date','open','adj_close', str(W)+'_'+str(k)+'_lowerBand', str(W)+'_'+str(k)+'_upperBand', str(W)+'_'+str(k)+'_signal'])
    
    
    profitSeries = pd.Series(dtype="float64")
    transactionTypeSeries = pd.Series(dtype="string")
    LongTransactionDaysSeries = pd.Series(dtype="int16")
    ShortTransactionDaysSeries = pd.Series(dtype="int16")  
    USD_value = 100.0
    BTC_value = 0
    Long = False
    Short = False
    shortValue = 0
    shortSharesCount = 0
    LongTransactionCount = 0
    ShortTransactionCount = 0
    LongTransactionStart = None
    ShortTransactionStart = None
    i=0
    
    while i < (len(df_window)-1):         
                        if (Long == False):
                            shares = (USD_value + BTC_value)/df_window.iloc[i,2]
                        elif Long == True:
                            BTC_value += (float(df_window.iloc[i,2])*float(shares))-float(BTC_value)
                        
                        if df_window.iloc[i,5] == 1:
                            #if the close value is below the lower bollinger band and you have no position, long
                            if Long == False and Short == False:
                                BTC_value = float(df_window.iloc[i,2]) * float(shares)
                                USD_value = USD_value - BTC_value
                                Long = True
                                LongTransactionCount += 1
                                boughtAt = BTC_value
                                LongTransactionStart = i
                            
                            #if the close value is below the lower bollinger band and you a short position, close the position
                            if Short == True:
                                #profitSeries.set_value(i, (shortValue * shortSharesCount) - float(df_window.iloc[i,2]) * shortSharesCount)
                                profitSeries.at[i] = (shortValue * shortSharesCount) - float(df_window.iloc[i,2]) * shortSharesCount
                                
                                #transactionTypeSeries.set_value(i, 'short')
                                transactionTypeSeries.at[i] = 'short'

                                
                                USD_value +=  (shortValue * shortSharesCount) - (float(df_window.iloc[i,2]) * shortSharesCount)
                                Short = False
                                #ShortTransactionDaysSeries.set_value(i, i - ShortTransactionStart)                                 
                                ShortTransactionDaysSeries.at[i] = i - ShortTransactionStart                              

                                
                        elif df_window.iloc[i,5] == -1:
                            #if the close value is above the upper bollinger band and you have a long position, close the long position
                            if Long == True:
                                #profitSeries.set_value(i,(BTC_value - boughtAt))
                                profitSeries.at[i] = (BTC_value - boughtAt)

                                
                                #transactionTypeSeries.set_value(i, 'long')
                                transactionTypeSeries.at[i] = 'long'

                                USD_value +=  BTC_value
                                BTC_value = 0
                                Long = False
                                #LongTransactionDaysSeries.set_value(i, i - LongTransactionStart)
                                LongTransactionDaysSeries.at[i] = i - LongTransactionStart 

                                
                            #if the close value is above the upper bollinger band and you don't have a position, short   
                            elif Short == False and Long == False:
                                shortSharesCount = float(shares)
                                shortValue = float(df_window.iloc[i,2])
                                Short = True
                                ShortTransactionCount += 1
                                ShortTransactionStart = i
                            
                        i += 1
                        
    
    
    df_profit = profitSeries.to_frame()
    df_profit.rename(columns={0:'profit'},inplace=True)
    df_profit['transaction_type'] = transactionTypeSeries.values
    df_longProfit = df_profit[df_profit['transaction_type']=='long']
    df_shortProfit = df_profit[df_profit['transaction_type']=='short']
    
 
    avg_long_profit = round(df_longProfit['profit'].mean(),4)
    avg_short_profit = round(df_shortProfit['profit'].mean(),4)
    
    avg_days_long = round(LongTransactionDaysSeries.mean(),1)
    avg_days_short = round(ShortTransactionDaysSeries.mean(),1)
    

    new_row = {'window':int(W), 'standard_deviation':float(k), 'average_trade_profit':round(df_profit['profit'].mean(),2),
               'longTransactionCount':LongTransactionCount, 'shortTransactionCount': ShortTransactionCount,
               'avg_long_profit':avg_long_profit, 'avg_short_profit':avg_short_profit,
               'avg_days_long':avg_days_long, 'avg_days_short':avg_days_short}
    return new_row

df_2017 = df[df.td_year.isin([2017])].reset_index()

df_2018 = df[df.td_year.isin([2018])].reset_index()

year1 = 2017


for w in windowSequence:
    for s in standardDeviationSequence:
        df_2017 = pd.concat([df_2017, bollinger(w,s,year1)], axis=1, join='inner')  
        print(str(year1) + ' ' + str(w) + ' day moving average - upper and lower bands added for ' + str(s) + ' standard deviations.')

      
for w in windowSequence:
    for s in standardDeviationSequence: 
        df_windowProfit2017 = df_windowProfit2017.append(strategy1(w,s,year1), ignore_index=True)
        print(str(w) + ' day moving average - trade profits added for ' + str(s) + ' standard deviations.')
        
year1 = 2018

for w in windowSequence:
    for s in standardDeviationSequence:
        df_2018 = pd.concat([df_2018, bollinger(w,s,year1)], axis=1, join='inner')  
        print(str(year1) + ' ' + str(w) + ' day moving average - upper and lower bands added for ' + str(s) + ' standard deviations.')

      
for w in windowSequence:
    for s in standardDeviationSequence: 
        df_windowProfit2018 = df_windowProfit2018.append(strategy1(w,s,year1), ignore_index=True)
        print(str(w) + ' day moving average - trade profits added for ' + str(s) + ' standard deviations.')
        

#Optimal combination of k (standard deviation) and W (moving average window) for year 1
optimalWindow = df_windowProfit2017[['average_trade_profit']].idxmax()[0]

print('\nBOLLINGER BANDS - Optimal combination of k (standard deviation) and W (moving average window)')
print('The highest profit per transaction combination of W and k for year 1 is ' + str(df_windowProfit2017.iloc[optimalWindow,1]) + '.')
print('The highest profit occurs where W = ' + str(df_windowProfit2017.iloc[optimalWindow,0]) 
        + ' and k = ' + str(df_windowProfit2017.iloc[optimalWindow,8]))

#Optimal combination of k (standard deviation) and W (moving average window) for year 2
optimalWindow = df_windowProfit2017[['average_trade_profit']].idxmax()[0]
        
print('\nBOLLINGER BANDS - Optimal combination of k (standard deviation) and W (moving average window)')
print('The highest profit per transaction for combinations of W and k for year 2 is ' + str(df_windowProfit2018.iloc[optimalWindow,1]) + '.')
print('The highest profit occurs where W = ' + str(df_windowProfit2018.iloc[optimalWindow,0]) 
        + ' and k = ' + str(df_windowProfit2018.iloc[optimalWindow,8]))

#Year 1 Profit
print('\nBOLLINGER BANDS - A plot has been created showing average profit for combinations of W,k in year 1')
x=np.array(df_windowProfit2017.window)
y=np.array(df_windowProfit2017.standard_deviation)
z=np.array(df_windowProfit2017.average_trade_profit)
plt.title('Average profit')
plt.xlabel('days used for moving average: W')
plt.ylabel('standard deviation for Bollinger Bands: k')
plt.scatter(x,y,s = (z+abs(df_windowProfit2017.average_trade_profit.min())+1))