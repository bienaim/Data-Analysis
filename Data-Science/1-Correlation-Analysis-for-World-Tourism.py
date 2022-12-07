#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:47:31 2019

@author: Abien
"""
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print('WHICH FEATURES IN THE DATATSET BEST CORRELATE TO TOURISM IN THE FOLLOWING YEAR?')
print('\n\nCreating a correlation matrix for all indicators in the dataset...')

df_data_pivot_2 = pd.read_csv("WDIData_reduced.csv")

#INDICATORS USED FOR THE CORRELATION ANALYSIS
# 'Year',
# 'Population density (people per sq. km of land area)', 
# 'Population growth (annual %)',
# 'Population in largest city',
# 'Population in the largest city (% of urban population)',
# 'Population in urban agglomerations of more than 1 million',
# 'Population, total',
# 'Portfolio equity, net inflows (BoP, current US$)',
# 'Portfolio investment, bonds (PPG + PNG) (NFL, current US$)',
# 'Portfolio investment, net (BoP, current US$)',
# 'Rural land area (sq. km)',
# 'Rural population',
# 'Rural population (% of total population)',
# 'Rural population growth (annual %)',
# 'Real effective exchange rate index (2010 = 100)',
# 'Urban land area (sq. km)',
# 'Urban population',
# 'Urban population (% of total population)',
# 'Urban population growth (annual %)',
# 'Unemployment, total (% of total labor force) (national estimate)',
# 'Taxes on goods and services (% of revenue)',
# 'Tax revenue (% of GDP)',
# 'Start-up procedures to register a business (number)',
# 'Self-employed, total (% of total employment) (modeled ILO estimate)',
# 'Portfolio investment, net (BoP, current US$)',
# 'Pump price for diesel fuel (US$ per liter)',
# 'Pump price for gasoline (US$ per liter)',
# 'Profit tax (% of commercial profits)',
# 'Official exchange rate (LCU per US$, period average)',
# 'New businesses registered (number)',
# 'Net foreign assets (current LCU)',
# 'Mobile cellular subscriptions (per 100 people)',
# 'Listed domestic companies, total',
# 'Lending interest rate (%)',
# 'Land area (sq. km)',
# 'Labor force, total',
# 'International tourism, expenditures (% of total imports)',
# 'International tourism, expenditures (current US$)',
# 'International tourism, expenditures for passenger transport items (current US$)',
# 'International tourism, expenditures for travel items (current US$)',
# 'International tourism, receipts (% of total exports)',
# 'International tourism, receipts (current US$)',
# 'International tourism, receipts for passenger transport items (current US$)',
# 'International tourism, receipts for travel items (current US$)',
# 'Human capital index (HCI) (scale 0-1)',
# 'Gross national expenditure (% of GDP)',
# 'Gross national expenditure (current US$)',
# 'GDP (current US$)',
# 'GDP growth (annual %)',
# 'GDP per capita (current US$)',
# 'Forest area (% of land area)',
# 'Forest area (sq. km)',
# 'Foreign direct investment, net (BoP, current US$)',
# 'Foreign direct investment, net inflows (% of GDP)',
# 'Expense (% of GDP)',
# 'Deposit interest rate (%)',
# 'Cost of business start-up procedures (% of GNI per capita)',
# 'Average precipitation in depth (mm per year)',
# 'Air transport, passengers carried'



tourismSeries = pd.Series(df_data_pivot_2[('International tourism, receipts (current US$)')])
GDPSeries = pd.Series(df_data_pivot_2[('GDP (current US$)')])

# Calculate tourism as a % of GDP
df_data_pivot_2[('International tourism, receipts (% of GDP)')] = (tourismSeries.values)/(GDPSeries.values)

i=0
#Series for next year's tourism as a percentage of GDP
Next_Year_Percentage_Series = pd.Series()
#Series for next year's tourism receipts
Next_Year_Tourism_Series = pd.Series()
#Series for next year's category for tourism as a percentage of GDP (red or green)
Next_Year_Percentage_Series_Category = pd.Series()
#Series for next year's category for tourism receipts (red or green)
Next_Year_Tourism_Series_Category = pd.Series()

#Create series for increases or decreases in tourism
while i < len(df_data_pivot_2)-1:
    if df_data_pivot_2['Country'][i] == df_data_pivot_2['Country'][i+1]:
        #Next_Year_Percentage_Series.set_value(i, df_data_pivot_2['International tourism, receipts (% of GDP)'][i+1])
        Next_Year_Percentage_Series.at[i] = df_data_pivot_2['International tourism, receipts (% of GDP)'][i+1]
        #Next_Year_Tourism_Series.set_value(i, df_data_pivot_2['International tourism, receipts (current US$)'][i+1])
        Next_Year_Tourism_Series.at[i] = df_data_pivot_2['International tourism, receipts (current US$)'][i+1]
        #Add red and green categories to the series for an increase or decrease in tourism
        if (df_data_pivot_2['International tourism, receipts (current US$)'][i+1] 
                > df_data_pivot_2['International tourism, receipts (current US$)'][i]):
            #Next_Year_Tourism_Series_Category.set_value(i, 'green')
            Next_Year_Tourism_Series_Category.at[i] = 'green'
        else:
            #Next_Year_Tourism_Series_Category.set_value(i, 'red')
            Next_Year_Tourism_Series_Category.at[i] = 'red'
            
        #Add red and green categories to the series for an increase or decrease in tourism as % of GDP
        if (df_data_pivot_2['International tourism, receipts (% of GDP)'][i+1]
                > df_data_pivot_2['International tourism, receipts (% of GDP)'][i]):
            #Next_Year_Percentage_Series_Category.set_value(i, 'green')
            Next_Year_Percentage_Series_Category.at[i] = 'green'
        else:
            #Next_Year_Percentage_Series_Category.set_value(i, 'red')
            Next_Year_Percentage_Series_Category.at[i] = 'red'
            
    else:
        # Next_Year_Percentage_Series.set_value(i, 0)
        # Next_Year_Tourism_Series.set_value(i, 0)
        # Next_Year_Tourism_Series_Category.set_value(i, 'red')
        # Next_Year_Percentage_Series_Category.set_value(i, 'red')
        
        Next_Year_Percentage_Series.at[i] = 0
        Next_Year_Tourism_Series.at[i] = 0
        Next_Year_Tourism_Series_Category.at[i] = 'red'
        Next_Year_Percentage_Series_Category.at[i] = 'red'
        
    i += 1

#Add 0 to the last year of data
# Next_Year_Percentage_Series.set_value(i, 0)
# Next_Year_Tourism_Series.set_value(i, 0)
# Next_Year_Tourism_Series_Category.set_value(i, 'red')
# Next_Year_Percentage_Series_Category.set_value(i, 'red')

Next_Year_Percentage_Series.at[i] = 0
Next_Year_Tourism_Series.at[i] = 0
Next_Year_Tourism_Series_Category.at[i] = 'red'
Next_Year_Percentage_Series_Category.at[i] = 'red'

#Combine these series with the dataset
df_data_pivot_2['Next Year International tourism, receipts (% of GDP)'] = Next_Year_Percentage_Series.values
df_data_pivot_2['Next Year International tourism, receipts (current US$)'] = Next_Year_Tourism_Series.values
df_data_pivot_2['Next Year International tourism, receipts (% of GDP) category'] = Next_Year_Percentage_Series_Category.values
df_data_pivot_2['Next Year International tourism, receipts (current US$) category'] = Next_Year_Tourism_Series_Category.values

#Print full correlation matrix
print('\n\nCORRELATION MATRIX')
f = plt.figure(figsize=(19, 15))
plt.matshow(df_data_pivot_2.corr(), fignum=f.number)
plt.xticks(range(df_data_pivot_2.shape[1]), df_data_pivot_2.columns, fontsize=10, rotation=90)
plt.yticks(range(df_data_pivot_2.shape[1]), df_data_pivot_2.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
print('A plot has been created containing a correlation matrix for all indicators.')

#Print sorted correlations for tourism as a % of GDP
print('\n\nCORRELATIONS TO NEXT YEAR\'S TOURISM AS A % OF GDP')
print(str(df_data_pivot_2.corrwith(df_data_pivot_2['International tourism, receipts (% of GDP)']).sort_values(ascending = False)))

#Print sorted correlations for tourism as a % of GDP
print('\n\nCORRELATIONS TO NEXT YEAR\'S TOURISM RECEIPTS (IN $USD)')
print(str(df_data_pivot_2.corrwith(df_data_pivot_2['Next Year International tourism, receipts (current US$)']).sort_values(ascending = False)))



df_classifier_data = df_data_pivot_2[['Country', 'Year', 'New businesses registered (number)',
                                      'Air transport, passengers carried',
                                      'Portfolio investment, bonds (PPG + PNG) (NFL, current US$)', 
                                      'Listed domestic companies, total', 'International tourism, receipts (current US$)',
                                      'Next Year International tourism, receipts (current US$)',
                                      'Next Year International tourism, receipts (current US$) category']]

print('\n\nCORRELATION - SELECTING A FEW INDICATORS POSITIVELY AND NEGATIVELY CORRELATED TO NEXT YEAR\'S TOURISM')
print('- New businesses registered (number)')
print('- Air transport, passengers carried')
print('- Portfolio investment, bonds')
print('- Listed domestic companies, total')

#remove 2017 data which doesn't contain next year data or labels
df_classifier_data = df_classifier_data[~df_classifier_data.Year.isin(['2017'])]

list = ['Low & middle income','Low income','Lower middle income', 'World', 'High income',
'OECD members','Post-demographic dividend','IDA & IBRD total','Low & middle income','Middle income',
'IBRD only','Late-demographic dividend','East Asia & Pacific','Upper middle income','Europe & Central Asia',
'North America','East Asia & Pacific (excluding high income)','East Asia & Pacific (IDA & IBRD countries)',
'European Union','Early-demographic dividend','Euro area','Lower middle income','Latin America & Caribbean',
'Latin America & the Caribbean (IDA & IBRD countries)','Middle East & North Africa',
'Latin America & Caribbean (excluding high income)','Europe & Central Asia (IDA & IBRD countries)','Arab World',
'Europe & Central Asia (excluding high income)','South Asia','South Asia (IDA & IBRD)','United Arab Emirates',
'Middle East & North Africa (IDA & IBRD countries)','Middle East & North Africa (excluding high income)',
'Sub-Saharan Africa','Sub-Saharan Africa (IDA & IBRD countries)','Sub-Saharan Africa (excluding high income)',
'Central Europe and the Baltics','Least developed countries: UN classification','Heavily indebted poor countries (HIPC)',
'Pre-demographic dividend','Fragile and conflict affected situations']
df_classifier_data = df_classifier_data[~df_classifier_data.Country.isin(list)]


#Fill Empty values with median values
#New_Business_median = df_classifier_data['New businesses registered (number)'].median()
#df_classifier_data['New businesses registered (number)'] = df_classifier_data['New businesses registered (number)'].fillna(New_Business_median)
#
#Air_Transport_median = df_classifier_data['Air transport, passengers carried'].median()
#df_classifier_data['Air transport, passengers carried'] = df_classifier_data['Air transport, passengers carried'].fillna(Air_Transport_median)
#
#Listed_Companies_median = df_classifier_data['Listed domestic companies, total'].median()
#df_classifier_data['Listed domestic companies, total'] = df_classifier_data['Listed domestic companies, total'].fillna(Listed_Companies_median)
#
#Bond_Investments_median = df_classifier_data['Portfolio investment, bonds (PPG + PNG) (NFL, current US$)'].median()
#df_classifier_data['Portfolio investment, bonds (PPG + PNG) (NFL, current US$)'] = df_classifier_data['Portfolio investment, bonds (PPG + PNG) (NFL, current US$)'].fillna(Bond_Investments_median)


#save files for subsequent analyses
df_classifier_data.to_csv("df_classifier_data.csv", index=False)
df_classifier_data_2016 = df_classifier_data[df_classifier_data['Year']==2016]
df_classifier_data_2016.to_csv("df_classifier_data_2016.csv", index=False)
print('\nTwo files have been created containing data needed for subsequent analyses.')