#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 00:12:30 2020

@author: Abien
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import numpy as np
import pandas as pd
import math
import time

imageData = []
showImageAnalysis = False

#Data collector
def getImageDifferences(model):
    imageTopDifference = [agent.imageDifferences for agent in model.schedule.agents]
    
    return imageTopDifference

#Image Agent    
class ImageAgent(Agent):
    """An agent with color values and color forecasts."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        
        #values representing pixels for each side of the image
        #generate 100 random pixel values for each side
        self.top = np.random.rand(100)*10
        self.bottom = np.random.rand(100)*10
        self.left = np.random.rand(100)*10
        self.right = np.random.rand(100)*10
        
        #estimates of values beyond the image border
        #generate 100 random pixel values for each side
        self.topForecast = np.random.rand(100)*10
        self.bottomForecast = np.random.rand(100)*10
        self.leftForecast = np.random.rand(100)*10
        self.rightForecast = np.random.rand(100)*10
        

        
        
        self.topDifference = 0.0
        self.bottomDifference = 0.0
        self.leftDifference = 0.0
        self.rightDifference = 0.0
        
        self.top_bottomOtherImage = 0
        self.bottom_topOtherImage = 0
        self.left_rightOtherImgage = 0
        self.right_leftOtherImage = 0
        
        self.originalPosition = ("","")
        
        #a list of images whose sides have already been compared
        self.imagesChecked = []
        
        #error values between each image's sides and the other image's sides
        self.imageDifferences = []
        
        print("IMAGE " + str(self.unique_id) + " VALUES")
        print("  Top:" + str(self.top) + "  Bottom:" + str(self.bottom) + "  Left:" + str(self.left) + "  Right:" + str(self.right))
        
    def step(self):
        # The agent's step will go here.
        self.move()
        self.compare()
        self.returnToOriginalPosition()
    
    def move(self):
        
        x=0
        y=0
        checked = False
        moved = False
        
        #print("Comparing image sides for differences and estimates...")
        while y < self.model.grid.height:
            if x < self.model.grid.width:
                if moved == False:
                    if (self.originalPosition != (x,y)):
                        
                        if not ((x,y) in self.imagesChecked):
                            checked = False
                            #print("\nCurrent Image: ",self.unique_id)

                            
                        if not (x,y) in self.imagesChecked:
                            new_position = (x,y)
                            self.model.grid.move_agent(self, new_position)
                            #print("moved to ",self.pos)
                            
                            moved = True
                            break
                        else:
                            x+=1
                    else:
                        x+=1
                
                else:
                    break
                
            else:
                y+=1
                x=0
                
    def compare(self):
        imagesInArea = self.model.grid.get_cell_list_contents([self.pos])
        if len(imagesInArea) > 1:
            if imagesInArea[0] == self:
                otherImage = imagesInArea[1]
            else:
                otherImage = imagesInArea[0]
            
            #Compare each side's forecast to the other image's adjacent side actual values
            #take the square of this difference
            topDifference = np.square(self.topForecast - otherImage.bottom).sum()
            bottomDifference = np.square(self.bottomForecast - otherImage.top).sum()
            leftDifference = np.square(self.leftForecast - otherImage.right).sum()
            rightDifference = np.square(self.rightForecast - otherImage.left).sum()
            
            self.top_bottomImage = otherImage.unique_id
            self.bottom_topImage = otherImage.unique_id
            self.left_rightImgage = otherImage.unique_id
            self.right_leftImage = otherImage.unique_id
            
            self.imageDifferences.append([self.unique_id,
                                          otherImage.unique_id,
                                          round(topDifference,2),
                                          round(bottomDifference,2),
                                          round(leftDifference,2),
                                          round(rightDifference,2)])
            imageData.append([self.unique_id,
                                          otherImage.unique_id,
                                          round(topDifference,2),
                                          round(bottomDifference,2),
                                          round(leftDifference,2),
                                          round(rightDifference,2)])
            
            #print("Image Comparisons and Errors\n",self.imageDifferences)

            #Add the image to the list of images checked
            self.imagesChecked.append(otherImage.pos)
            
            #print("home ",self.originalPosition)
            #print("Checked ",otherImage.pos)
            #print("Compared to ", self.imagesChecked)
            
    def returnToOriginalPosition(self):
        self.model.grid.move_agent(self, self.originalPosition)
        #print("Returned to ",self.originalPosition)
        
#Image Arrangement Model
class ImageSpace(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        
        print("\n\nThe model contains " + str(N) + " images ")
        print("\nCREATING IMAGES AND A SEQUENCE OF VALUES FOR EACH SIDE OF EACH IMAGE\n")
        for i in range(self.num_agents):
            a = ImageAgent(i, self)
            self.schedule.add(a)
            
            pos = self.grid.find_empty()
            
            self.grid.place_agent(a, pos)
            a.originalPosition = pos
            
        self.datacollector = DataCollector(
                model_reporters={"Differences": getImageDifferences},
                agent_reporters={"Differences":"imageDifferences"})
            
    def step(self):
        self.datacollector.collect(self)
        """Advance the model by one step."""
        self.schedule.step()

gridSide = int(input("\nWhat is the length of each side for the model grid?\n (sides should be greater than 1 and ideally less than 10) \n\n"))


#Get the start time
t1 = time.perf_counter()
        
global totalImages        
global gridWidth
global gridHeight
        
totalImages = int(math.pow(int(gridSide),2))
gridWidth = gridSide
gridHeight = gridSide
        
model = ImageSpace(totalImages,gridWidth,gridHeight)

agent_counts = np.zeros((model.grid.width, model.grid.height))

#run the model for as many steps as there are images
for i in range(int(totalImages)):
    model.step()
    
#the model data collector
modelImageData = model.datacollector.get_model_vars_dataframe()
#print("\nModel Data collector")
#print(modelImageData)
#the agent data collector
agentImageData = model.datacollector.get_agent_vars_dataframe()
#print("\nAgent Data collector")
#print(agentImageData.head(16))
#place datacollector information into a dataframe
df = pd.DataFrame(imageData,columns=['image1','image2','top','bottom','left','right'])
print("\n\nDATAFRAME CONTAINING ERRORS BETWEEN ADJACENT EDGES FOR ALL IMAGE PAIRS")
print(df)

#stop the timer
t2 = time.perf_counter()

continueApplication = input("\n\nPRESS Y TO ARRANGE THE IMAGES USING HEURISTIC-BASED SEARCH\n\n")

while continueApplication == "Yes" or "yes" or "Y" or "y":

    break

#Plan and Search functions
#Is the current image the best match for the next image? If not, return the actual best image and its error.
def isNextImageBestImage(image, side, bestSearchNumber):

    bestImage = df[df['image2']==int(image)].sort_values(by=side).head(totalImages).iloc[bestSearchNumber-1]['image1']
    bestError = df[df['image2']==int(image)].sort_values(by=side).head(totalImages).iloc[bestSearchNumber-1][side]
    
    print("Image "+str(int(bestImage))+" is a better "+str(side)+" source image for image "+str(int(image)))
    
    return int(bestImage), bestError


#Search for the next image based on current image, side, and attempt number           
def findNextImage2(image, side, searchNumber,searchRank,listOfImagesUsed):
        
    nextImage = df[df['image1']==image].sort_values(by=side).head(totalImages).iloc[searchNumber-1]['image2']
    error = df[df['image1']==image].sort_values(by=side).head(totalImages).iloc[searchNumber-1][side]
    
    print("Image "+str(int(image))+" has image "+str(int(nextImage))+" as its next best "+str(side)+" target image")
    
    bestImageSearchAttempt = searchRank
    
    bestImage, bestError = isNextImageBestImage(int(nextImage), side, bestImageSearchAttempt)

   #If the best image has already been used, search for the next best image
    while (int(bestImage) in listOfImagesUsed) and (int(bestImage) != int(nextImage) and (int(bestImage) != image)):
        print("Image "+str(int(bestImage))," has already been used")
        bestImageSearchAttempt +=1
        bestImage, bestError = isNextImageBestImage(int(nextImage), side, bestImageSearchAttempt)
        
    
    return int(nextImage), error, bestImage, bestError

#Arrange images from left to right, row by row     
def arrange2(startingImage):
            
    df3Data = [[startingImage,(0,0),0]]
    df_3 = pd.DataFrame(df3Data, columns=['image','position','error'])
     
    imagesUsed = [startingImage]
    y=0
    x=1
    
    while y < gridHeight:
        print("y: ",y)
        if x < gridWidth:
            print("x: ",x)
            attempts = 1
            searchRank = 1
            nextImage, error, bestImage, bestError = findNextImage2(int(imagesUsed[-1]),'right',attempts,searchRank,imagesUsed)
            
            #if the selected image has already been used, or the current image is not the best image, search for another image
            #if on the other hand, the selected image has not been used, or the error is less than or equal to the best error, break and continue
            while (nextImage in imagesUsed)  or (error > bestError):
                print("Image "+str(int(nextImage))," has already been used or isn't the best image to use.")
                if attempts == totalImages-1:
                    attempts = 0
                    searchRank += 1
                attempts+=1
                nextImage, error,bestImage, bestError = findNextImage2(int(imagesUsed[-1]),'right',attempts,searchRank,imagesUsed)
                
            imagesUsed.append(nextImage)
            newRow = pd.DataFrame([[nextImage,(x,y),error]], columns=['image','position','error'])
            df_3 = df_3.append(newRow)
            print("\nImage Arrangement\n")
            print(df_3)
            
            if (y == gridHeight-1 and x==gridWidth-1):
                break
            
            x+=1        
        elif (x == gridWidth):
            attempts=1
            searchRank = 1
            nextImage, error, bestImage, bestError = findNextImage2(int(imagesUsed[-gridWidth]),'top',attempts,searchRank,imagesUsed)
            
            #if the best image has already been placed on the grid
            #select the next best image
            
            while (nextImage in imagesUsed) or ((bestImage not in imagesUsed) and (error > bestError)):
                if attempts == totalImages-1:
                    attempts = 0
                    searchRank += 1
                attempts+=1
                nextImage, error, bestImage, bestError = findNextImage2(int(imagesUsed[-gridWidth]),'top',attempts,searchRank,imagesUsed)       
                
            imagesUsed.append(nextImage)
            newRow = pd.DataFrame([[nextImage,(0,y+1),error]], columns=['image','position','error'])
            df_3 = df_3.append(newRow)
            print("\nImage Arrangement\n")
            print(df_3)
            
            x=1
            y+=1
            
    return imagesUsed, df_3

#Find the best first image of the sequence
def isFirstImage(imageNumber):
     left = df[df['image1']==imageNumber]['left'].sum()
     bottom = df[df['image1']==imageNumber]['left'].sum()
            
     return float(left + bottom)

#Greedy Search functions    
#Search for the next image based on current image, side, and attempt number           
def findNextImage(image, side, searchNumber):
    nextImage = df[df['image1']==image].sort_values(by=side).head(totalImages).iloc[searchNumber-1]['image2']
    error = df[df['image1']==image].sort_values(by=side).head(totalImages).iloc[searchNumber-1][side]
    return int(nextImage), error

#Arrange images from left to right, row by row     
def arrange(startingImage):
            
    df3Data = [[startingImage,(0,0),0]]
    df_3 = pd.DataFrame(df3Data, columns=['image','position','error'])
     
    imagesUsed = [startingImage]
    y=0
    x=1
    
    while y < gridHeight:
        print("y: ",y)
        if x < gridWidth:
            print("x: ",x)
            attempts = 1
            nextImage, error = findNextImage(int(imagesUsed[-1]), 'right', attempts)
                
            while nextImage in imagesUsed:
                attempts+=1
                nextImage, error = findNextImage(int(imagesUsed[-1]), 'right',attempts)
                
            imagesUsed.append(nextImage)
            newRow = pd.DataFrame([[nextImage,(x,y),error]], columns=['image','position','error'])
            df_3 = df_3.append(newRow)
            print("\nImage Arrangement\n")
            print(df_3)
            
            if (y == gridHeight-1 and x==gridWidth-1):
                break
            
            x+=1        
        elif (x == gridWidth):
            attempts=1
            nextImage, error = findNextImage(int(imagesUsed[-gridWidth]), 'top', attempts)
            
            #if the best image has already been placed on the grid
            #select the next best image
            while nextImage in imagesUsed:
                attempts+=1
                nextImage, error = findNextImage(int(imagesUsed[-gridWidth]), 'top',attempts)
                
                
            imagesUsed.append(nextImage)
            newRow = pd.DataFrame([[nextImage,(0,y+1),error]], columns=['image','position','error'])
            df_3 = df_3.append(newRow)
            print("\nImage Arrangement\n")
            print(df_3)
            
            x=1
            y+=1
            
    return imagesUsed, df_3


firstImageData = []
firstImageErrorPercent = []
ErrorSum = 0.0

for i in range(int(totalImages)):
    firstImageData.append(isFirstImage(i))
    ErrorSum += firstImageData[i]
    #print(ErrorSum)
    
#print(firstImageData)
print('\nTotal Error for all images is ' + str(ErrorSum))

i = 0
while i < totalImages:
    firstImageErrorPercent.append(round(isFirstImage(i)/ErrorSum,4))
    i+=1
    
Df_data = {'Total Image Error':firstImageData, '% Total Error from all Images': firstImageErrorPercent}
firstImageDf = pd.DataFrame(Df_data)
firstImage = int(firstImageDf.sort_values(by='% Total Error from all Images',ascending=False).head(1).index[0])
firstImageLikelihood = firstImageDf.sort_values(by='% Total Error from all Images',ascending=False).head(1).iloc[0]['% Total Error from all Images']

print('\n')
print(firstImageDf)
print('\n')

print("\nThe estimated likelihood of Image " + str(firstImage) + ' being the first image is ' + str(firstImageLikelihood))

print('\nImage ' + str(firstImage) + ' seems to be a good choice for the first image\n')

#start the timer
t3 = time.perf_counter()

sequence1, resultsDataframe1 = arrange2(firstImage)

#stop the timer
t4 = time.perf_counter()

continueApplication = input("\n\nPRESS Y TO ARRANGE THE IMAGES USING GREEDY BEST-FIRST SEARCH\n\n")

while continueApplication == "Yes" or "yes" or "Y" or "y":

    break

#start the timer
t5 = time.perf_counter()

#sequence, resultsDataframe = arrange(totalImages-1)
sequence2, resultsDataframe2 = arrange(firstImage)

#stop the timer
t6 = time.perf_counter()


print("\n")

print(sequence1)
print("\nThe Heuristic-Based Search sequence was completed in " + str(round((t4-t3)+(t2-t1),2)) + " seconds")
print('\nThe total error between image edges was '+str(resultsDataframe1['error'].sum()))
#print the likelihood of the image being first

print("\n")

print(sequence2)
print("\nThe Greedy Search sequence was completed in " + str(round((t6-t5)+(t2-t1),2)) + " seconds")
print('\nThe total error between image edges was '+str(resultsDataframe2['error'].sum()))