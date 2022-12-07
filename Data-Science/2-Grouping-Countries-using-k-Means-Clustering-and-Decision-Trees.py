#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:10:02 2019

@author: Abien
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fill empty values with 0. Alternatively, empty values can be filled with
# the median value from the category by uncommenting lines 206 - 216 from
# the python file named "1-Correlation-Analysis-for-World-Tourism.py"
df_kMeans = pd.read_csv("df_classifier_data_2016.csv").fillna(0)



id_data = np.arange(len(df_kMeans))
New_Businesses_Registered = np.array(df_kMeans['New businesses registered (number)'])
Bond_Investments = np.array(df_kMeans['Portfolio investment, bonds (PPG + PNG) (NFL, current US$)'])
Air_Transport = np.array(df_kMeans['Air transport, passengers carried'])
Label = np.array(df_kMeans['Next Year International tourism, receipts (current US$) category'])

data = pd.DataFrame(
        {'id':  id_data,
        'Label': Label,
        'New_Businesses': New_Businesses_Registered,
        'Bond_Investments': Bond_Investments,
        'Air_Transport': Air_Transport},
         columns = ['id', 'New_Businesses', 'Bond_Investments',
                      'Air_Transport', 'Label'])

init_centers = np.array([[5.0,7.0],[5.5,9.0]])
colmap = {0: 'orange', 1: 'blue', 2: 'green', 3: 'red'}
x = data[['New_Businesses', 'Bond_Investments', 'Air_Transport']].values

#scaler = StandardScaler()
#scaler.fit(x)
#x = scaler.transform(x) 
print('WHICH COUNTRIES ARE SIMILAR IN TERMS OF ANNUAL NEW BUSINESSES, BOND INVESTMENTS, AND AIR TRANSPORT?')


#print distortion and find optimal clusters, k
print('\n\nk-MEANS CLUSTERING - CREATING CLUSTERS, FINDING CENTROIDS AND DISTORTION VALUES...')
inertia_list = []
for k in range(1 ,9):
       kmeans_classifier = KMeans(n_clusters=k)
       y_kmeans = kmeans_classifier.fit_predict(x)
       inertia = kmeans_classifier.inertia_
       print('Distortion for ' + str(k) + ' clusters is ' + str(inertia))
       inertia_list.append(inertia)


fig,ax = plt.subplots(1,figsize =(7,5))
plt.plot(range(1, 9), inertia_list, marker='o',
           color='green')
plt.legend()
plt.title('OPTIMAL NUMBER OF CLUSTERS TO USE FOR GROUPING BY SIMILARITY')
plt.xlabel('number of clusters: k')
plt.ylabel('distortion')
plt.tight_layout()
plt.show()

print('\n\nk-MEANS CLUSTERING - PLOT CREATED FOR CLUSTERS AND DISTORTION')
print('Visual Analysis of the distortion plot shows that about')
print('3 clusters will minimize error (distortion) while maintaining')
print('meaningfully distinct groups.\n')

n_clusters = 3
kmeans_classifier = KMeans(n_clusters = n_clusters, init='random')
y_means = kmeans_classifier.fit_predict(x)

centroids = kmeans_classifier.cluster_centers_

clusterSeries = pd.Series(y_means)

#add cluster to data    
data['cluster']=clusterSeries.values
#add country to data
data['Country']=pd.Series(df_kMeans['Country']).values

#plot data across three dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range (n_clusters):
       new_df = data[data['cluster']==i]
       ax.scatter(new_df['New_Businesses'], new_df['Bond_Investments'], new_df['Air_Transport'], color=colmap[i],
                   s=10 , label='points in cluster' + str(i+1))
for i in range (n_clusters):
       ax.scatter(centroids[i][0], centroids[i][1], color=colmap[i],
                   marker='x', s=1000 , label='centroid' + str(i+1))
for i in range (len(data)):
       x_text = data['New_Businesses'].iloc[i] + 0.05
       y_text = data['Bond_Investments'].iloc[i] + 0.05
       z_text = data['Air_Transport'].iloc[i] + 0.2
       id_text = data['id'].iloc[i]

ax.legend (loc='upper left')
plt.xlim(0, 100000)
#plt.ylim(5, 15)
ax.set_xlabel('New_Businesses')
ax.set_ylabel('Bond_Investments')
ax.set_zlabel('Air_Transport')
plt.show()
print('\n\nk-MEANS CLUSTERING - PLOT CREATED TO SHOW GROUPS AND CLUSTER CENTERS')
print('Visual analysis of the cluster plot shows how the groups')
print('might be defined based on the three selected dimensions.')


#print clusters
print('\n\nCLUSTER 1 COUNTRIES: ')
print(data[data['cluster']==0]['Country'])
print('\nCLUSTER 2 COUNTRIES: ')
print(data[data['cluster']==1]['Country'])
print('\nCLUSTER 3 COUNTRIES: ')
print(data[data['cluster']==2]['Country'])


feature_names = ['New_Businesses', 'Bond_Investments', 'Air_Transport']

df = pd.DataFrame(data, columns=feature_names)
df['target'] = data.cluster


X = data[feature_names].values
le = LabelEncoder()
Y = le.fit_transform(data['cluster'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle = True)
tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
tree_classifier.fit(X_train,Y_train)

prediction = tree_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print('\n\n\nDECISION TREES - DESCRIBING CLUSTERS USING A DECISION TREE')

fn=['New_Businesses', 'Bond_Investments', 'Air_Transport']
cn=['0', '1', '2']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(tree_classifier,
               feature_names = fn, 
               class_names=cn,
               filled = True);

print('\nA plot has been created containing the decision tree and rules describing each cluster.')

print('\nBond Investments appears to be a major factor for defining the groups.')

