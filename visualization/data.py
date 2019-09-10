# This file contains the Data extractor and visualization class

import numpy as np
import pandas as pd
import os
import shutil

class Data:

	# Class variables >>>>

	# Initial data path
	initial_data_path = os.path.join("visualization","student","initial","student-por.csv")
	
	# NaN value & directory path
	nan_directory_path = os.path.join("visualization","student","nan")
	nan_data_path = os.path.join("visualization","student","nan","student-por.csv")

	# NaN value & directory path
	cleaned_directory_path = os.path.join("visualization","student","cleaned")
	cleaned_data_path = os.path.join("visualization","student","cleaned","student-por-postclean.csv")
	
	# Percentage NaN
	percentage_nan = [0.03, 0.015, 0.010] # The variable to keep track of what percent of data is modified to NaN

	# Columns list 
	column_list1 = ['Pstatus','Medu','Fedu','Mjob','Fjob','reason','failures','romantic','famrel','Dalc','Walc']
	column_list2 = ['famsize','traveltime','studytime','goout','health','famsup']
	column_list3 = ['internet','freetime','guardian']

	#Dividing the columns based on what operation needs to be applied
	column_mean = ['Medu','Fedu','Mjob','Fjob','famrel','freetime','goout','health','traveltime']
	column_median = ['famsize','studytime','famsup','failures','Dalc','Walc']
	column_mode = ['internet','guardian','Pstatus','reason','romantic']


	# Init method
	def __init__(self):
		pass

# Introducing NaNs

	# Function to add NaN
	def add_nan(self,df,column_list,percentage):

		# Adding NaN
		for col in column_list:
		    ori_rat = df[col].isna().mean()

		    if ori_rat >= percentage: continue

		    add_miss_rat = (percentage - ori_rat) / (1 - ori_rat)
		    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
		    df.loc[vals_to_nan, col] = np.NaN



	# Function to introduce NaN values
	def introduce_nan(self):
	
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))
		
		# Calling add_nan function to modify 3 lists of columns
		self.add_nan(df,self.column_list1,self.percentage_nan[0])
		self.add_nan(df,self.column_list2,self.percentage_nan[1])
		self.add_nan(df,self.column_list3,self.percentage_nan[2])

		# Filling NaN
		df.fillna("NaN", inplace = True)

		# Saving the new dataset

		if os.path.exists(self.nan_directory_path):
			shutil.rmtree(self.nan_directory_path)

		os.mkdir(self.nan_directory_path) # making new directory to save newversion
		df.to_csv(self.nan_data_path)
        
#Replacing NaNs
    
    #Functions to replace NaNs
	def rep_NaN_mean(self,df,col):
		df.fillna(df.mean()[col])
        
	def rep_NaN_median(self,df,col):
		df.fillna(df.median()[col])
        
	def rep_NaN_mode(self,df,col):
		df.fillna(df.mode()[col])
        
    #Function to remove the NaNs
	def replace_nan(self):
        
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.nan_data_path))
        
        #Calling rem_NaN functions 
		self.rep_NaN_mean(df,self.column_mean)
		self.rep_NaN_median(df,self.column_median)
		self.rep_NaN_mode(df,self.column_mode)
        
    	# Saving the new dataset
		if os.path.exists(self.cleaned_directory_path):
			shutil.rmtree(self.cleaned_directory_path)

		os.mkdir(self.cleaned_directory_path) # making new directory to save new version
		df.to_csv(self.cleaned_data_path)
		print("done") 






