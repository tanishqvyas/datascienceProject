# This file contains the Data extractor and visualization class

import numpy as np
import pandas as pd
import os
import shutil
import math

class Data:

	# Class variables >>>>

	# Initial data path
	initial_directory_path = os.path.join("visualization","student","initial")
	initial_data_path = os.path.join(initial_directory_path,"student-por.csv")
	
	# NaN value & directory path
	nan_directory_path = os.path.join("visualization","student","nan")
	nan_data_path = os.path.join(nan_directory_path,"student-por.csv")

	# NaN value & directory path
	cleaned_directory_path = os.path.join("visualization","student","cleaned")
	cleaned_data_path = os.path.join(cleaned_directory_path,"student-por-postclean.csv")
	
	# Percentage NaN
	percentage_nan = [0.03, 0.015, 0.010] # The variable to keep track of what percent of data is modified to NaN

	# Columns list 
	column_list1 = ['Pstatus','Medu','Fedu','Mjob','Fjob','reason','failures','romantic','famrel','Dalc','Walc']
	column_list2 = ['famsize','traveltime','studytime','goout','health','famsup']
	column_list3 = ['internet','freetime','guardian']

	#Dividing the columns based on what operation needs to be applied
	column_mean = ['Medu','Fedu','famrel','freetime','goout','health','traveltime']
	column_median = ['studytime','famsup','failures','Dalc','Walc']
	column_mode = ['famsize','internet','guardian','Pstatus','reason','romantic','Fjob','Mjob']





	# Init method
	def __init__(self):
		self.boolean_filter()


	# Function to save modified dataset as a new version of DataFrame
	def save_file(self, df, directory_path, data_path):

		if os.path.exists(directory_path):
			shutil.rmtree(directory_path)

		os.mkdir(directory_path) # making new directory to save newversion
		df.to_csv(data_path,index = False)
		print("done")

	# Function to replace yes with 1 and no with 0 respectively
	def boolean_filter(self):

		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))

		# Replacing values
		df.replace(to_replace ="yes", value = 1, inplace = True)
		df.replace(to_replace ="no", value = 0, inplace = True)

		# saving file
		self.save_file(df, self.initial_directory_path, self.initial_data_path)


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
		self.save_file(df, self.nan_directory_path, self.nan_data_path)
        
#Replacing NaNs
    
    #Functions to replace NaNs
	def rep_NaN_mean(self,df,col):
		df.fillna(df.mean()[col].round(0), inplace = True)
        
	def rep_NaN_median(self,df,col):
		df.fillna(df.median()[col].round(0), inplace = True)
        
	def rep_NaN_mode(self,df,col):
		df.fillna(df.mode()[col], inplace = True)
        
    # Function to remove the NaNs
	def replace_nan(self):
        
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.nan_data_path))
        
        #Calling rem_NaN functions 
		self.rep_NaN_mean(df,self.column_mean)
		self.rep_NaN_median(df,self.column_median)
		self.rep_NaN_mode(df,self.column_mode)
        
    	# Saving the new dataset
		self.save_file(df, self.cleaned_directory_path, self.cleaned_data_path)

	# Function to return Q1 Q2 or Q3
	def get_quartile_value(self,data_list, n, quartile_ratio):

		if (n+1)*quartile_ratio == int((n+1)*quartile_ratio):
			return data_list[(n+1)*quartile_ratio]

		else:
			return ( data_list[math.ceil((n+1)*quartile_ratio)] + data_list[math.floor((n+1)*quartile_ratio)] ) / 2

	# Function to find median
	def get_IQR(self, data_list):

		# finding number of elements
		num_of_elements = len(data_list)

		# fetching Q1, Q2 & Q3 values
		Q1 = get_quartile_value(data_list,num_of_elements,0.25)
		Q3 = get_quartile_value(data_list,num_of_elements,0.75)

		return Q3 - Q1


	# Function to find bin_size using Freedman-Diaconis formula
	def get_bin_size(self,data_list, n):

		bin_size = (2 * get_IQR(data_list)) / (math.pow(n, 1/3))
		return math.ceil(bin_size)

	# Function to calculate num of classes
	def get_count_classes(self, data_list):

		num_of_classes = (max(data_list)-min(data_list)) / self.get_bin_size(data_list, len(data_list))
		return math.ceil(num_of_classes)


	# Graph plotters






