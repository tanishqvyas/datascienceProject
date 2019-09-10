# This file contains the Data extractor and visualization class

import numpy as np
import pandas as pd
import os

class Data:

	# Class variables >>>>

	# Initial data path
	initial_data_path = os.path.join("visualization","student","initial","student-por.csv")
	
	# NaN value & directory path
	nan_directory_path = os.path.join("visualization","student","nan")
	nan_data_path = os.path.join("visualization","student","nan","student-por.csv")
	
	# Percentage NaN
	percentage_nan = [0.03, 0.015, 0.010] # The variable to keep track of what percent of data is modified to NaN

	# Columns list 

	column_list1 = ['Pstatus','Medu','Fedu','Mjob','Fjob','reason','paid','romantic','famrel','Dalc']
	column_list2 = ['famsize','traveltime','studytime','goout','health','famsup']
	column_list3 = ['activities','internet','freetime','guardian']



	# Init method
	def __init__(self):
		pass

	# Function to introduce NaN values
	def introduce_nan(self):
	
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))
		
		# Adding NaN
		for col in self.column_list1:
		    ori_rat = df[col].isna().mean()

		    if ori_rat >= self.percentage_nan[0]: continue

		    add_miss_rat = (self.percentage_nan[0] - ori_rat) / (1 - ori_rat)
		    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
		    df.loc[vals_to_nan, col] = np.NaN

		for col in self.column_list2:
		    ori_rat = df[col].isna().mean()

		    if ori_rat >= self.percentage_nan[1]: continue

		    add_miss_rat = (self.percentage_nan[1] - ori_rat) / (1 - ori_rat)
		    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
		    df.loc[vals_to_nan, col] = np.NaN

		for col in self.column_list3:
		    ori_rat = df[col].isna().mean()

		    if ori_rat >= self.percentage_nan[2]: continue

		    add_miss_rat = (self.percentage_nan[2] - ori_rat) / (1 - ori_rat)
		    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
		    df.loc[vals_to_nan, col] = np.NaN

		# Filling NaN
		df.fillna("NaN", inplace = True)

		# Saving the new dataset
		os.mkdir(self.nan_directory_path) # making new directory to save newversion
		df.to_csv(self.nan_data_path)
		print("done") 






