#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:40:13 2019

@author: Abien
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


print('LINEAR REGRESSION - If we use the past W days of prices to forecast the next day\'s price,')
print('then take long or short trading positions (based on if the forecast is for an increase or decrease respectively)')
print('what number of days used for forecasting results in the highest returns?\n\n')

#import file and create data frames
df = pd.read_csv("BTC-USD.csv")
#data frames for year 1 and year 2 profit data
df_windowProfit2017 = pd.DataFrame(columns=['window','average_trade_profit','longTransactionCount','shortTransactionCount',
                                            'avg_long_profit','avg_short_profit','avg_days_long', 'avg_days_short'])
df_windowProfit2018 = pd.DataFrame(columns=['window','average_trade_profit','longTransactionCount','shortTransactionCount',
                                            'avg_long_profit','avg_short_profit','avg_days_long', 'avg_days_short'])

new_row = None

#function to return forecast, signal, R square value given W and year
def window(windowsize,year1):
    df_year = df[df.td_year.isin([year1])]
    regressionSeries = pd.Series()
    signalSeries = pd.Series()
    r_squareSeries = pd.Series()
    
    W = windowsize
    df_length = (len(df_year))
    i=0
    regressionSeries.append(pd.Series(np.arange(W)))
    
    while i < df_length:
        if i < W:            
            #regressionSeries.set_value(i,0)
            regressionSeries.at[i] = 0

            #signalSeries.set_value(i,0)
            signalSeries.at[i] = 0
            
                
            #r_squareSeries.set_value(i,0)
            r_squareSeries.at[i] = 0
            
            
            i += 1
        else:
            W_values = np.array(df_year.iloc[i-W:i,12].values)
            x= np.arange(W)
            y= W_values
            
            degree = 1
            weights = np.polyfit(x,y, degree)
            
            model = np.poly1d(weights)
            
            y_predicted = []
            for x in range(0, W):
                y_predicted.append(model(x))
                
            #r_squareSeries.set_value(i,r2_score(y,y_predicted))
            r_squareSeries.at[i] = r2_score(y,y_predicted)

                        
            #regressionSeries.set_value(i, model(W))
            regressionSeries.at[i] = model(W)

            
            if regressionSeries[i]>df_year.iloc[i,12]:
                #signalSeries.set_value(i,1)
                signalSeries.at[i] = 1

            elif regressionSeries[i]<df_year.iloc[i,12]:
                #signalSeries.set_value(i,-1)
                signalSeries.at[i] = -1

            else:
                #signalSeries.set_value(i,0)
                signalSeries.at[i] = 0

            
            i+= 1
            
    window_data = pd.concat([regressionSeries, signalSeries, r_squareSeries],axis=1)
    window_data = window_data.rename(columns={0:str(W)+'_forecast', 1:str(W)+'_signal', 2:str(W)+'_r_square'})
    return window_data


# function to return profit/loss and long/short position statistics given W and year
def strategy1(window,year1):
    W = window
    if year1 == 2017:
        df_window = pd.DataFrame(df_2017, columns = ['trade_date','open','adj_close',str(W)+'_forecast',str(W)+'_signal'])
    elif year1 == 2018:
        df_window = pd.DataFrame(df_2018, columns = ['trade_date','open','adj_close',str(W)+'_forecast',str(W)+'_signal'])
    
    
    profitSeries = pd.Series()
    transactionTypeSeries = pd.Series()
    LongTransactionDaysSeries = pd.Series()
    ShortTransactionDaysSeries = pd.Series()  
    starting_value = 100.0
    USD_value = starting_value
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
                        
                        if df_window.iloc[i,4] == 1:
                            #if signal is 1 and you don't have a position, long
                            if Long == False and Short == False:
                                BTC_value = float(df_window.iloc[i,2]) * float(shares)
                                USD_value = USD_value - BTC_value
                                Long = True
                                LongTransactionCount += 1
                                boughtAt = BTC_value
                                LongTransactionStart = i
                            
                            #if signal is 1 and you have a short position, close the position
                            if Short == True:
                                #profitSeries.set_value(i, (shortValue * shortSharesCount) - float(df_window.iloc[i,2]) * shortSharesCount)
                                profitSeries.at[i] = (shortValue * shortSharesCount) - float(df_window.iloc[i,2]) * shortSharesCount

                                #transactionTypeSeries.set_value(i, 'short')
                                transactionTypeSeries.at[i] = 'short'

                                
                                USD_value +=  (shortValue * shortSharesCount) - (float(df_window.iloc[i,2]) * shortSharesCount)
                                Short = False
                                #ShortTransactionDaysSeries.set_value(i, i - ShortTransactionStart)
                                ShortTransactionDaysSeries.at[i] = i - ShortTransactionStart
 
                                
                        elif df_window.iloc[i,4] == -1:
                            #if signal is -1 and you have a long position, close the position
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

                                
                            #if signal is -1 and you don't have a position, short    
                            elif Short == False and Long == False:
                                shortSharesCount = float(shares)
                                shortValue = float(df_window.iloc[i,2])
                                Short = True
                                ShortTransactionCount += 1
                                ShortTransactionStart = i
                        
                            
                        i += 1
                        

    print('\n' + str(W) + ' Day Model - starting with $' + str(starting_value) + ' and using a "long and short" trading strategy:')                    
    print('At the end of the year, the value would be $' + str(round(USD_value,2)) + ' and $' + str(round(BTC_value))+' of BTC\n')
    
    
    df_profit = profitSeries.to_frame()
    df_profit.rename(columns={0:'profit'},inplace=True)
    df_profit['transaction_type'] = transactionTypeSeries.values
    df_longProfit = df_profit[df_profit['transaction_type']=='long']
    df_shortProfit = df_profit[df_profit['transaction_type']=='short']
    

 
    avg_long_profit = round(df_longProfit['profit'].mean(),4)
    avg_short_profit = round(df_shortProfit['profit'].mean(),4)
    
    avg_days_long = round(LongTransactionDaysSeries.mean(),1)
    avg_days_short = round(ShortTransactionDaysSeries.mean(),1)
    
    #return profit/loss data and long/short position statistics
    new_row = {'window':int(W), 'average_trade_profit':round(df_profit['profit'].mean(),2),
               'longTransactionCount':LongTransactionCount, 'shortTransactionCount': ShortTransactionCount,
               'avg_long_profit':avg_long_profit, 'avg_short_profit':avg_short_profit,
               'avg_days_long':avg_days_long, 'avg_days_short':avg_days_short}
    return new_row
 

#Linear Regression 1 - OPTIMAL NUMBER OF DAYS, W TO USE FOR MODEL
minWindow = 5
maxWindow = 20
sequence = list(range(minWindow,maxWindow+1))    
df_2017 = df[df.td_year.isin([2017])].reset_index()

for w in sequence:  
    df_2017 = pd.concat([df_2017, window(w,2017)], axis=1, join='inner')  
    print(str(w) + ' days of data has been used to create next-day forecasts and R square values')
      

for w in sequence:  
    df_windowProfit2017 = df_windowProfit2017.append(strategy1(w,2017), ignore_index=True)


print('\n\nLINEAR REGRESSION - OPTIMAL NUMBER OF DAYS, W TO USE FOR MODEL')
print('The optimal value of W is ' + str(int(df_windowProfit2017.loc[df_windowProfit2017['average_trade_profit'].idxmax()][0])))
fig = df_windowProfit2017.plot(kind='bar',x='window',y='average_trade_profit',color='green', 
                               title = 'AVERAGE TRADE PROFIT BY # OF DAYS USED').get_figure()
fig.savefig('Average_Trade_Profit.pdf')


#Linear Regression 2 - USING THE OPTIMAL YEAR 1 MODEL FOR YEAR 2'
print('\n\nLINEAR REGRESSION - USING THE OPTIMAL YEAR 1 MODEL FOR YEAR 2')
#find index of largest average trade profit
optimalWindow = int(df_windowProfit2017.loc[df_windowProfit2017['average_trade_profit'].idxmax()][0])
minWindow = optimalWindow
maxWindow = optimalWindow
sequence = list(range(minWindow,maxWindow+1))    

df_2018 = df[df.td_year.isin([2018])].reset_index()

for w in sequence:  
    df_2018 = pd.concat([df_2018, window(w,2018)], axis=1, join='inner')  
    print('The optimal linear regression window, W from the first year is ' + str(w) + ' days of data')
    print(str(w) + ' days of data has been used to create next-day forecasts and R square values for the second year')
    


print('\nThe average r squared for year 2 is ' + str(round(df_2018[str(optimalWindow)+'_r_square'].mean(),4)))
print('On average, the previous ' + str(w) + ' days of prices explains roughly ' + 
       str(round(df_2018[str(optimalWindow)+'_r_square'].mean(),2)*100) + '% of variation in the next day\'s prices')    
fig = df_2018.plot(kind='line',x='index',y= str(optimalWindow) + "_" + 'r_square',color='blue', title = 'R squares for year 2').get_figure()
fig.savefig('R_Square_for_Year_2.pdf')


#Linear Regression 3 - NUMBER OF LONG AND SHORT TRADES IN YEAR 2
for w in sequence:
    df_windowProfit2018 = df_windowProfit2018.append(strategy1(w,2018), ignore_index= True)
    #print(str(w) + ' day window average trade profit added')

print('\n\nLINEAR REGRESSION - NUMBER OF LONG AND SHORT TRADES IN YEAR 2')
print('In year 2, there were ' + str(int(df_windowProfit2018.iloc[0,2])) + ' long transactions and '
      + str(int(df_windowProfit2018.iloc[0,3])) + ' short transactions.')

#Linear Regression 4 - OUTCOME OF LONG AND SHORT TRADES IN YEAR 2
print('\n\nLINEAR REGRESSION - OUTCOME OF LONG AND SHORT TRADES IN YEAR 2')
print('The average profit per long position in year 2 is ' + str(df_windowProfit2018.iloc[0,4]))
print('The average profit per short position in year 2 is ' + str(df_windowProfit2018.iloc[0,5]))

#Linear Regression 5 - AVERAGE NUMBER OF DAYS IN LONG AND SHORT POSITIONS FOR YEAR 2
print('\n\nLINEAR REGRESSION - AVERAGE NUMBER OF DAYS IN LONG AND SHORT POSITIONS FOR YEAR 2')
print('The average number of days for a long position in year 2 is ' + str(df_windowProfit2018.iloc[0,6]))
print('The average number of days for a short position in year 2 is ' + str(df_windowProfit2018.iloc[0,7]))

#Linear Regression 6 - AVERAGE NUMBER OF DAYS IN LONG AND SHORT POSITIONS FOR YEAR 1
print('\n\nLINEAR REGRESSION - AVERAGE NUMBER OF DAYS IN LONG AND SHORT POSITIONS FOR YEAR 1')
print('The average number of days for a long position in year 1 is ' + str(df_windowProfit2017.loc[df_windowProfit2017['average_trade_profit'].idxmax()][6]))
print('The average number of days for a short position in year 1 is ' + str(df_windowProfit2017.loc[df_windowProfit2017['average_trade_profit'].idxmax()][7]))
